from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
@dataclass
class _Logger:
    log_dir: T.Optional[str] = None
    log_depth: T.List[str] = field(default_factory=list)
    log_file: T.Optional[T.TextIO] = None
    log_timestamp_start: T.Optional[float] = None
    log_fatal_warnings = False
    log_disable_stdout = False
    log_errors_only = False
    logged_once: T.Set[T.Tuple[str, ...]] = field(default_factory=set)
    log_warnings_counter = 0
    log_pager: T.Optional['subprocess.Popen'] = None
    _LOG_FNAME: T.ClassVar[str] = 'meson-log.txt'

    @contextmanager
    def no_logging(self) -> T.Iterator[None]:
        self.log_disable_stdout = True
        try:
            yield
        finally:
            self.log_disable_stdout = False

    @contextmanager
    def force_logging(self) -> T.Iterator[None]:
        restore = self.log_disable_stdout
        self.log_disable_stdout = False
        try:
            yield
        finally:
            self.log_disable_stdout = restore

    def set_quiet(self) -> None:
        self.log_errors_only = True

    def set_verbose(self) -> None:
        self.log_errors_only = False

    def set_timestamp_start(self, start: float) -> None:
        self.log_timestamp_start = start

    def shutdown(self) -> T.Optional[str]:
        if self.log_file is not None:
            path = self.log_file.name
            exception_around_goer = self.log_file
            self.log_file = None
            exception_around_goer.close()
            return path
        self.stop_pager()
        return None

    def start_pager(self) -> None:
        if not colorize_console():
            return
        pager_cmd = []
        if 'PAGER' in os.environ:
            pager_cmd = shlex.split(os.environ['PAGER'])
        else:
            less = shutil.which('less')
            if not less and is_windows():
                git = shutil.which('git')
                if git:
                    path = Path(git).parents[1] / 'usr' / 'bin'
                    less = shutil.which('less', path=str(path))
            if less:
                pager_cmd = [less]
        if not pager_cmd:
            return
        try:
            env = os.environ.copy()
            if 'LESS' not in env:
                env['LESS'] = 'RXF'
            if 'LV' not in env:
                env['LV'] = '-c'
            self.log_pager = subprocess.Popen(pager_cmd, stdin=subprocess.PIPE, text=True, encoding='utf-8', env=env)
        except Exception as e:
            if 'PAGER' in os.environ:
                from .mesonlib import MesonException
                raise MesonException(f'Failed to start pager: {str(e)}')

    def stop_pager(self) -> None:
        if self.log_pager:
            try:
                self.log_pager.stdin.flush()
                self.log_pager.stdin.close()
            except OSError:
                pass
            self.log_pager.wait()
            self.log_pager = None

    def initialize(self, logdir: str, fatal_warnings: bool=False) -> None:
        self.log_dir = logdir
        self.log_file = open(os.path.join(logdir, self._LOG_FNAME), 'w', encoding='utf-8')
        self.log_fatal_warnings = fatal_warnings

    def process_markup(self, args: T.Sequence[TV_Loggable], keep: bool, display_timestamp: bool=True) -> T.List[str]:
        arr: T.List[str] = []
        if self.log_timestamp_start is not None and display_timestamp:
            arr = ['[{:.3f}]'.format(time.monotonic() - self.log_timestamp_start)]
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, str):
                arr.append(arg)
            elif isinstance(arg, AnsiDecorator):
                arr.append(arg.get_text(keep))
            else:
                arr.append(str(arg))
        return arr

    def force_print(self, *args: str, nested: bool, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
        if self.log_disable_stdout:
            return
        iostr = io.StringIO()
        print(*args, sep=sep, end=end, file=iostr)
        raw = iostr.getvalue()
        if self.log_depth:
            prepend = self.log_depth[-1] + '| ' if nested else ''
            lines = []
            for l in raw.split('\n'):
                l = l.strip()
                lines.append(prepend + l if l else '')
            raw = '\n'.join(lines)
        try:
            output = self.log_pager.stdin if self.log_pager else None
            print(raw, end='', file=output)
        except UnicodeEncodeError:
            cleaned = raw.encode('ascii', 'replace').decode('ascii')
            print(cleaned, end='')

    def debug(self, *args: TV_Loggable, sep: T.Optional[str]=None, end: T.Optional[str]=None, display_timestamp: bool=True) -> None:
        arr = process_markup(args, False, display_timestamp)
        if self.log_file is not None:
            print(*arr, file=self.log_file, sep=sep, end=end)
            self.log_file.flush()

    def _log(self, *args: TV_Loggable, is_error: bool=False, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None, display_timestamp: bool=True) -> None:
        arr = process_markup(args, False, display_timestamp)
        if self.log_file is not None:
            print(*arr, file=self.log_file, sep=sep, end=end)
            self.log_file.flush()
        if colorize_console():
            arr = process_markup(args, True, display_timestamp)
        if not self.log_errors_only or is_error:
            force_print(*arr, nested=nested, sep=sep, end=end)

    def _debug_log_cmd(self, cmd: str, args: T.List[str]) -> None:
        if not _in_ci:
            return
        args = [f'"{x}"' for x in args]
        self.debug('!meson_ci!/{} {}'.format(cmd, ' '.join(args)))

    def cmd_ci_include(self, file: str) -> None:
        self._debug_log_cmd('ci_include', [file])

    def log(self, *args: TV_Loggable, is_error: bool=False, once: bool=False, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None, display_timestamp: bool=True) -> None:
        if once:
            self._log_once(*args, is_error=is_error, nested=nested, sep=sep, end=end, display_timestamp=display_timestamp)
        else:
            self._log(*args, is_error=is_error, nested=nested, sep=sep, end=end, display_timestamp=display_timestamp)

    def log_timestamp(self, *args: TV_Loggable) -> None:
        if self.log_timestamp_start:
            self.log(*args)

    def _log_once(self, *args: TV_Loggable, is_error: bool=False, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None, display_timestamp: bool=True) -> None:
        """Log variant that only prints a given message one time per meson invocation.

        This considers ansi decorated values by the values they wrap without
        regard for the AnsiDecorator itself.
        """

        def to_str(x: TV_Loggable) -> str:
            if isinstance(x, str):
                return x
            if isinstance(x, AnsiDecorator):
                return x.text
            return str(x)
        t = tuple((to_str(a) for a in args))
        if t in self.logged_once:
            return
        self.logged_once.add(t)
        self._log(*args, is_error=is_error, nested=nested, sep=sep, end=end, display_timestamp=display_timestamp)

    def _log_error(self, severity: _Severity, *rargs: TV_Loggable, once: bool=False, fatal: bool=True, location: T.Optional[BaseNode]=None, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None, is_error: bool=True) -> None:
        from .mesonlib import MesonException, relpath
        if severity is _Severity.NOTICE:
            label: TV_LoggableList = [bold('NOTICE:')]
        elif severity is _Severity.WARNING:
            label = [yellow('WARNING:')]
        elif severity is _Severity.ERROR:
            label = [red('ERROR:')]
        elif severity is _Severity.DEPRECATION:
            label = [red('DEPRECATION:')]
        args = label + list(rargs)
        if location is not None:
            location_file = relpath(location.filename, os.getcwd())
            location_str = get_error_location_string(location_file, location.lineno)
            location_list = T.cast('TV_LoggableList', [location_str])
            args = location_list + args
        log(*args, once=once, nested=nested, sep=sep, end=end, is_error=is_error)
        self.log_warnings_counter += 1
        if self.log_fatal_warnings and fatal:
            raise MesonException('Fatal warnings enabled, aborting')

    def error(self, *args: TV_Loggable, once: bool=False, fatal: bool=True, location: T.Optional[BaseNode]=None, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
        return self._log_error(_Severity.ERROR, *args, once=once, fatal=fatal, location=location, nested=nested, sep=sep, end=end, is_error=True)

    def warning(self, *args: TV_Loggable, once: bool=False, fatal: bool=True, location: T.Optional[BaseNode]=None, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
        return self._log_error(_Severity.WARNING, *args, once=once, fatal=fatal, location=location, nested=nested, sep=sep, end=end, is_error=True)

    def deprecation(self, *args: TV_Loggable, once: bool=False, fatal: bool=True, location: T.Optional[BaseNode]=None, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
        return self._log_error(_Severity.DEPRECATION, *args, once=once, fatal=fatal, location=location, nested=nested, sep=sep, end=end, is_error=True)

    def notice(self, *args: TV_Loggable, once: bool=False, fatal: bool=True, location: T.Optional[BaseNode]=None, nested: bool=True, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
        return self._log_error(_Severity.NOTICE, *args, once=once, fatal=fatal, location=location, nested=nested, sep=sep, end=end, is_error=False)

    def exception(self, e: Exception, prefix: T.Optional[AnsiDecorator]=None) -> None:
        if prefix is None:
            prefix = red('ERROR:')
        self.log()
        args: T.List[T.Union[AnsiDecorator, str]] = []
        if all((getattr(e, a, None) is not None for a in ['file', 'lineno', 'colno'])):
            path = get_relative_path(Path(e.file), Path(os.getcwd()))
            args.append(f'{path}:{e.lineno}:{e.colno}:')
        if prefix:
            args.append(prefix)
        args.append(str(e))
        with self.force_logging():
            self.log(*args, is_error=True)

    @contextmanager
    def nested(self, name: str='') -> T.Generator[None, None, None]:
        self.log_depth.append(name)
        try:
            yield
        finally:
            self.log_depth.pop()

    def get_log_dir(self) -> str:
        return self.log_dir

    def get_log_depth(self) -> int:
        return len(self.log_depth)

    @contextmanager
    def nested_warnings(self) -> T.Iterator[None]:
        old = self.log_warnings_counter
        self.log_warnings_counter = 0
        try:
            yield
        finally:
            self.log_warnings_counter = old

    def get_warning_count(self) -> int:
        return self.log_warnings_counter