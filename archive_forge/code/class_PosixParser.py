import os
import shlex
import subprocess
class PosixParser:
    """
    The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.
    """

    @staticmethod
    def join(argv):
        return ' '.join((quote(arg) for arg in argv))

    @staticmethod
    def split(cmd):
        return shlex.split(cmd, posix=True)