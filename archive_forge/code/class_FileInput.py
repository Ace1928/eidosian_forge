import io
import sys, os
from types import GenericAlias
class FileInput:
    """FileInput([files[, inplace[, backup]]], *, mode=None, openhook=None)

    Class FileInput is the implementation of the module; its methods
    filename(), lineno(), fileline(), isfirstline(), isstdin(), fileno(),
    nextfile() and close() correspond to the functions of the same name
    in the module.
    In addition it has a readline() method which returns the next
    input line, and a __getitem__() method which implements the
    sequence behavior. The sequence must be accessed in strictly
    sequential order; random access and readline() cannot be mixed.
    """

    def __init__(self, files=None, inplace=False, backup='', *, mode='r', openhook=None, encoding=None, errors=None):
        if isinstance(files, str):
            files = (files,)
        elif isinstance(files, os.PathLike):
            files = (os.fspath(files),)
        else:
            if files is None:
                files = sys.argv[1:]
            if not files:
                files = ('-',)
            else:
                files = tuple(files)
        self._files = files
        self._inplace = inplace
        self._backup = backup
        self._savestdout = None
        self._output = None
        self._filename = None
        self._startlineno = 0
        self._filelineno = 0
        self._file = None
        self._isstdin = False
        self._backupfilename = None
        self._encoding = encoding
        self._errors = errors
        if sys.flags.warn_default_encoding and 'b' not in mode and (encoding is None) and (openhook is None):
            import warnings
            warnings.warn("'encoding' argument not specified.", EncodingWarning, 2)
        if mode not in ('r', 'rb'):
            raise ValueError("FileInput opening mode must be 'r' or 'rb'")
        self._mode = mode
        self._write_mode = mode.replace('r', 'w')
        if openhook:
            if inplace:
                raise ValueError('FileInput cannot use an opening hook in inplace mode')
            if not callable(openhook):
                raise ValueError('FileInput openhook must be callable')
        self._openhook = openhook

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.nextfile()
        finally:
            self._files = ()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self._readline()
            if line:
                self._filelineno += 1
                return line
            if not self._file:
                raise StopIteration
            self.nextfile()

    def nextfile(self):
        savestdout = self._savestdout
        self._savestdout = None
        if savestdout:
            sys.stdout = savestdout
        output = self._output
        self._output = None
        try:
            if output:
                output.close()
        finally:
            file = self._file
            self._file = None
            try:
                del self._readline
            except AttributeError:
                pass
            try:
                if file and (not self._isstdin):
                    file.close()
            finally:
                backupfilename = self._backupfilename
                self._backupfilename = None
                if backupfilename and (not self._backup):
                    try:
                        os.unlink(backupfilename)
                    except OSError:
                        pass
                self._isstdin = False

    def readline(self):
        while True:
            line = self._readline()
            if line:
                self._filelineno += 1
                return line
            if not self._file:
                return line
            self.nextfile()

    def _readline(self):
        if not self._files:
            if 'b' in self._mode:
                return b''
            else:
                return ''
        self._filename = self._files[0]
        self._files = self._files[1:]
        self._startlineno = self.lineno()
        self._filelineno = 0
        self._file = None
        self._isstdin = False
        self._backupfilename = 0
        if 'b' not in self._mode:
            encoding = self._encoding or 'locale'
        else:
            encoding = None
        if self._filename == '-':
            self._filename = '<stdin>'
            if 'b' in self._mode:
                self._file = getattr(sys.stdin, 'buffer', sys.stdin)
            else:
                self._file = sys.stdin
            self._isstdin = True
        elif self._inplace:
            self._backupfilename = os.fspath(self._filename) + (self._backup or '.bak')
            try:
                os.unlink(self._backupfilename)
            except OSError:
                pass
            os.rename(self._filename, self._backupfilename)
            self._file = open(self._backupfilename, self._mode, encoding=encoding, errors=self._errors)
            try:
                perm = os.fstat(self._file.fileno()).st_mode
            except OSError:
                self._output = open(self._filename, self._write_mode, encoding=encoding, errors=self._errors)
            else:
                mode = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                if hasattr(os, 'O_BINARY'):
                    mode |= os.O_BINARY
                fd = os.open(self._filename, mode, perm)
                self._output = os.fdopen(fd, self._write_mode, encoding=encoding, errors=self._errors)
                try:
                    os.chmod(self._filename, perm)
                except OSError:
                    pass
            self._savestdout = sys.stdout
            sys.stdout = self._output
        elif self._openhook:
            if self._encoding is None:
                self._file = self._openhook(self._filename, self._mode)
            else:
                self._file = self._openhook(self._filename, self._mode, encoding=self._encoding, errors=self._errors)
        else:
            self._file = open(self._filename, self._mode, encoding=encoding, errors=self._errors)
        self._readline = self._file.readline
        return self._readline()

    def filename(self):
        return self._filename

    def lineno(self):
        return self._startlineno + self._filelineno

    def filelineno(self):
        return self._filelineno

    def fileno(self):
        if self._file:
            try:
                return self._file.fileno()
            except ValueError:
                return -1
        else:
            return -1

    def isfirstline(self):
        return self._filelineno == 1

    def isstdin(self):
        return self._isstdin
    __class_getitem__ = classmethod(GenericAlias)