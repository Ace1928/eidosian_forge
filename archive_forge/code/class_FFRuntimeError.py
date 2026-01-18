import errno
import shlex
import subprocess
class FFRuntimeError(Exception):
    """Raise when FFmpeg/FFprobe command line execution returns a non-zero exit code.

    The resulting exception object will contain the attributes relates to command line execution:
    ``cmd``, ``exit_code``, ``stdout``, ``stderr``.
    """

    def __init__(self, cmd, exit_code, stdout, stderr):
        self.cmd = cmd
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        message = '`{0}` exited with status {1}\n\nSTDOUT:\n{2}\n\nSTDERR:\n{3}'.format(self.cmd, exit_code, (stdout or b'').decode(), (stderr or b'').decode())
        super(FFRuntimeError, self).__init__(message)