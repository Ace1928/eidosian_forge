import time
from paste.wsgilib import catch_errors
class _ProgressFile(object):
    """
    This is the input-file wrapper used to record the number of
    ``paste.bytes_received`` for the given request.
    """

    def __init__(self, environ, rfile):
        self._ProgressFile_environ = environ
        self._ProgressFile_rfile = rfile
        self.flush = rfile.flush
        self.write = rfile.write
        self.writelines = rfile.writelines

    def __iter__(self):
        environ = self._ProgressFile_environ
        riter = iter(self._ProgressFile_rfile)

        def iterwrap():
            for chunk in riter:
                environ[ENVIRON_RECEIVED] += len(chunk)
                yield chunk
        return iter(iterwrap)

    def read(self, size=-1):
        chunk = self._ProgressFile_rfile.read(size)
        self._ProgressFile_environ[ENVIRON_RECEIVED] += len(chunk)
        return chunk

    def readline(self):
        chunk = self._ProgressFile_rfile.readline()
        self._ProgressFile_environ[ENVIRON_RECEIVED] += len(chunk)
        return chunk

    def readlines(self, hint=None):
        chunk = self._ProgressFile_rfile.readlines(hint)
        self._ProgressFile_environ[ENVIRON_RECEIVED] += len(chunk)
        return chunk