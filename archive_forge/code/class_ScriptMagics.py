import asyncio
import asyncio.exceptions
import atexit
import errno
import os
import signal
import sys
import time
from subprocess import CalledProcessError
from threading import Thread
from traitlets import Any, Dict, List, default
from IPython.core import magic_arguments
from IPython.core.async_helpers import _AsyncIOProxy
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.utils.process import arg_split
@magics_class
class ScriptMagics(Magics):
    """Magics for talking to scripts
    
    This defines a base `%%script` cell magic for running a cell
    with a program in a subprocess, and registers a few top-level
    magics that call %%script with common interpreters.
    """
    event_loop = Any(help="\n        The event loop on which to run subprocesses\n\n        Not the main event loop,\n        because we want to be able to make blocking calls\n        and have certain requirements we don't want to impose on the main loop.\n        ")
    script_magics: List = List(help="Extra script cell magics to define\n        \n        This generates simple wrappers of `%%script foo` as `%%foo`.\n        \n        If you want to add script magics that aren't on your path,\n        specify them in script_paths\n        ").tag(config=True)

    @default('script_magics')
    def _script_magics_default(self):
        """default to a common list of programs"""
        defaults = ['sh', 'bash', 'perl', 'ruby', 'python', 'python2', 'python3', 'pypy']
        if os.name == 'nt':
            defaults.extend(['cmd'])
        return defaults
    script_paths = Dict(help="Dict mapping short 'ruby' names to full paths, such as '/opt/secret/bin/ruby'\n        \n        Only necessary for items in script_magics where the default path will not\n        find the right interpreter.\n        ").tag(config=True)

    def __init__(self, shell=None):
        super(ScriptMagics, self).__init__(shell=shell)
        self._generate_script_magics()
        self.bg_processes = []
        atexit.register(self.kill_bg_processes)

    def __del__(self):
        self.kill_bg_processes()

    def _generate_script_magics(self):
        cell_magics = self.magics['cell']
        for name in self.script_magics:
            cell_magics[name] = self._make_script_magic(name)

    def _make_script_magic(self, name):
        """make a named magic, that calls %%script with a particular program"""
        script = self.script_paths.get(name, name)

        @magic_arguments.magic_arguments()
        @script_args
        def named_script_magic(line, cell):
            if line:
                line = '%s %s' % (script, line)
            else:
                line = script
            return self.shebang(line, cell)
        named_script_magic.__doc__ = '%%{name} script magic\n        \n        Run cells with {script} in a subprocess.\n        \n        This is a shortcut for `%%script {script}`\n        '.format(**locals())
        return named_script_magic

    @magic_arguments.magic_arguments()
    @script_args
    @cell_magic('script')
    def shebang(self, line, cell):
        """Run a cell via a shell command

        The `%%script` line is like the #! line of script,
        specifying a program (bash, perl, ruby, etc.) with which to run.

        The rest of the cell is run by that program.

        Examples
        --------
        ::

            In [1]: %%script bash
               ...: for i in 1 2 3; do
               ...:   echo $i
               ...: done
            1
            2
            3
        """
        if self.event_loop is None:
            if sys.platform == 'win32':
                event_loop = asyncio.WindowsProactorEventLoopPolicy().new_event_loop()
            else:
                event_loop = asyncio.new_event_loop()
            self.event_loop = event_loop
            asyncio_thread = Thread(target=event_loop.run_forever, daemon=True)
            asyncio_thread.start()
        else:
            event_loop = self.event_loop

        def in_thread(coro):
            """Call a coroutine on the asyncio thread"""
            return asyncio.run_coroutine_threadsafe(coro, event_loop).result()

        async def _readchunk(stream):
            try:
                return await stream.readuntil(b'\n')
            except asyncio.exceptions.IncompleteReadError as e:
                return e.partial
            except asyncio.exceptions.LimitOverrunError as e:
                return await stream.read(e.consumed)

        async def _handle_stream(stream, stream_arg, file_object):
            while True:
                chunk = (await _readchunk(stream)).decode('utf8', errors='replace')
                if not chunk:
                    break
                if stream_arg:
                    self.shell.user_ns[stream_arg] = chunk
                else:
                    file_object.write(chunk)
                    file_object.flush()

        async def _stream_communicate(process, cell):
            process.stdin.write(cell)
            process.stdin.close()
            stdout_task = asyncio.create_task(_handle_stream(process.stdout, args.out, sys.stdout))
            stderr_task = asyncio.create_task(_handle_stream(process.stderr, args.err, sys.stderr))
            await asyncio.wait([stdout_task, stderr_task])
            await process.wait()
        argv = arg_split(line, posix=not sys.platform.startswith('win'))
        args, cmd = self.shebang.parser.parse_known_args(argv)
        try:
            p = in_thread(asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE))
        except OSError as e:
            if e.errno == errno.ENOENT:
                print("Couldn't find program: %r" % cmd[0])
                return
            else:
                raise
        if not cell.endswith('\n'):
            cell += '\n'
        cell = cell.encode('utf8', 'replace')
        if args.bg:
            self.bg_processes.append(p)
            self._gc_bg_processes()
            to_close = []
            if args.out:
                self.shell.user_ns[args.out] = _AsyncIOProxy(p.stdout, event_loop)
            else:
                to_close.append(p.stdout)
            if args.err:
                self.shell.user_ns[args.err] = _AsyncIOProxy(p.stderr, event_loop)
            else:
                to_close.append(p.stderr)
            event_loop.call_soon_threadsafe(lambda: asyncio.Task(self._run_script(p, cell, to_close)))
            if args.proc:
                proc_proxy = _AsyncIOProxy(p, event_loop)
                proc_proxy.stdout = _AsyncIOProxy(p.stdout, event_loop)
                proc_proxy.stderr = _AsyncIOProxy(p.stderr, event_loop)
                self.shell.user_ns[args.proc] = proc_proxy
            return
        try:
            in_thread(_stream_communicate(p, cell))
        except KeyboardInterrupt:
            try:
                p.send_signal(signal.SIGINT)
                in_thread(asyncio.wait_for(p.wait(), timeout=0.1))
                if p.returncode is not None:
                    print('Process is interrupted.')
                    return
                p.terminate()
                in_thread(asyncio.wait_for(p.wait(), timeout=0.1))
                if p.returncode is not None:
                    print('Process is terminated.')
                    return
                p.kill()
                print('Process is killed.')
            except OSError:
                pass
            except Exception as e:
                print('Error while terminating subprocess (pid=%i): %s' % (p.pid, e))
            return
        if args.raise_error and p.returncode != 0:
            rc = p.returncode or -9
            raise CalledProcessError(rc, cell)
    shebang.__skip_doctest__ = os.name != 'posix'

    async def _run_script(self, p, cell, to_close):
        """callback for running the script in the background"""
        p.stdin.write(cell)
        await p.stdin.drain()
        p.stdin.close()
        await p.stdin.wait_closed()
        await p.wait()
        for s in to_close:
            await s.read()
        self._gc_bg_processes()

    @line_magic('killbgscripts')
    def killbgscripts(self, _nouse_=''):
        """Kill all BG processes started by %%script and its family."""
        self.kill_bg_processes()
        print('All background processes were killed.')

    def kill_bg_processes(self):
        """Kill all BG processes which are still running."""
        if not self.bg_processes:
            return
        for p in self.bg_processes:
            if p.returncode is None:
                try:
                    p.send_signal(signal.SIGINT)
                except:
                    pass
        time.sleep(0.1)
        self._gc_bg_processes()
        if not self.bg_processes:
            return
        for p in self.bg_processes:
            if p.returncode is None:
                try:
                    p.terminate()
                except:
                    pass
        time.sleep(0.1)
        self._gc_bg_processes()
        if not self.bg_processes:
            return
        for p in self.bg_processes:
            if p.returncode is None:
                try:
                    p.kill()
                except:
                    pass
        self._gc_bg_processes()

    def _gc_bg_processes(self):
        self.bg_processes = [p for p in self.bg_processes if p.returncode is None]