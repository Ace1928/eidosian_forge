import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class TestFileEnvironment(object):
    """
    This represents an environment in which files will be written, and
    scripts will be run.
    """
    __test__ = False

    def __init__(self, base_path, template_path=None, script_path=None, environ=None, cwd=None, start_clear=True, ignore_paths=None, ignore_hidden=True):
        """
        Creates an environment.  ``base_path`` is used as the current
        working directory, and generally where changes are looked for.

        ``template_path`` is the directory to look for *template*
        files, which are files you'll explicitly add to the
        environment.  This is done with ``.writefile()``.

        ``script_path`` is the PATH for finding executables.  Usually
        grabbed from ``$PATH``.

        ``environ`` is the operating system environment,
        ``os.environ`` if not given.

        ``cwd`` is the working directory, ``base_path`` by default.

        If ``start_clear`` is true (default) then the ``base_path``
        will be cleared (all files deleted) when an instance is
        created.  You can also use ``.clear()`` to clear the files.

        ``ignore_paths`` is a set of specific filenames that should be
        ignored when created in the environment.  ``ignore_hidden``
        means, if true (default) that filenames and directories
        starting with ``'.'`` will be ignored.
        """
        self.base_path = base_path
        self.template_path = template_path
        if environ is None:
            environ = os.environ.copy()
        self.environ = environ
        if script_path is None:
            if sys.platform == 'win32':
                script_path = environ.get('PATH', '').split(';')
            else:
                script_path = environ.get('PATH', '').split(':')
        self.script_path = script_path
        if cwd is None:
            cwd = base_path
        self.cwd = cwd
        if start_clear:
            self.clear()
        elif not os.path.exists(base_path):
            os.makedirs(base_path)
        self.ignore_paths = ignore_paths or []
        self.ignore_hidden = ignore_hidden

    def run(self, script, *args, **kw):
        """
        Run the command, with the given arguments.  The ``script``
        argument can have space-separated arguments, or you can use
        the positional arguments.

        Keywords allowed are:

        ``expect_error``: (default False)
            Don't raise an exception in case of errors
        ``expect_stderr``: (default ``expect_error``)
            Don't raise an exception if anything is printed to stderr
        ``stdin``: (default ``""``)
            Input to the script
        ``printresult``: (default True)
            Print the result after running
        ``cwd``: (default ``self.cwd``)
            The working directory to run in

        Returns a `ProcResponse
        <class-paste.fixture.ProcResponse.html>`_ object.
        """
        __tracebackhide__ = True
        expect_error = _popget(kw, 'expect_error', False)
        expect_stderr = _popget(kw, 'expect_stderr', expect_error)
        cwd = _popget(kw, 'cwd', self.cwd)
        stdin = _popget(kw, 'stdin', None)
        printresult = _popget(kw, 'printresult', True)
        args = list(map(str, args))
        assert not kw, 'Arguments not expected: %s' % ', '.join(kw.keys())
        if ' ' in script:
            assert not args, 'You cannot give a multi-argument script (%r) and arguments (%s)' % (script, args)
            script, args = script.split(None, 1)
            args = shlex.split(args)
        script = self._find_exe(script)
        all = [script] + args
        files_before = self._find_files()
        proc = subprocess.Popen(all, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=cwd, env=self.environ)
        stdout, stderr = proc.communicate(stdin)
        files_after = self._find_files()
        result = ProcResult(self, all, stdin, stdout, stderr, returncode=proc.returncode, files_before=files_before, files_after=files_after)
        if printresult:
            print(result)
            print('-' * 40)
        if not expect_error:
            result.assert_no_error()
        if not expect_stderr:
            result.assert_no_stderr()
        return result

    def _find_exe(self, script_name):
        if self.script_path is None:
            script_name = os.path.join(self.cwd, script_name)
            if not os.path.exists(script_name):
                raise OSError('Script %s does not exist' % script_name)
            return script_name
        for path in self.script_path:
            fn = os.path.join(path, script_name)
            if os.path.exists(fn):
                return fn
        raise OSError('Script %s could not be found in %s' % (script_name, ':'.join(self.script_path)))

    def _find_files(self):
        result = {}
        for fn in os.listdir(self.base_path):
            if self._ignore_file(fn):
                continue
            self._find_traverse(fn, result)
        return result

    def _ignore_file(self, fn):
        if fn in self.ignore_paths:
            return True
        if self.ignore_hidden and os.path.basename(fn).startswith('.'):
            return True
        return False

    def _find_traverse(self, path, result):
        full = os.path.join(self.base_path, path)
        if os.path.isdir(full):
            result[path] = FoundDir(self.base_path, path)
            for fn in os.listdir(full):
                fn = os.path.join(path, fn)
                if self._ignore_file(fn):
                    continue
                self._find_traverse(fn, result)
        else:
            result[path] = FoundFile(self.base_path, path)

    def clear(self):
        """
        Delete all the files in the base directory.
        """
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        os.mkdir(self.base_path)

    def writefile(self, path, content=None, frompath=None):
        """
        Write a file to the given path.  If ``content`` is given then
        that text is written, otherwise the file in ``frompath`` is
        used.  ``frompath`` is relative to ``self.template_path``
        """
        full = os.path.join(self.base_path, path)
        if not os.path.exists(os.path.dirname(full)):
            os.makedirs(os.path.dirname(full))
        f = open(full, 'wb')
        if content is not None:
            f.write(content)
        if frompath is not None:
            if self.template_path:
                frompath = os.path.join(self.template_path, frompath)
            f2 = open(frompath, 'rb')
            f.write(f2.read())
            f2.close()
        f.close()
        return FoundFile(self.base_path, path)