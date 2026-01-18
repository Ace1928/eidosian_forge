import os
import sys
import time
from rdkit import RDConfig
class OutputRedirectC:
    """Context manager which uses low-level file descriptors to suppress
  output to stdout/stderr, optionally redirecting to the named file(s).

  Suppress all output
  with Silence():
    <code>

  Redirect stdout to file
  with OutputRedirectC(stdout='output.txt', mode='w'):
    <code>

  Redirect stderr to file
  with OutputRedirectC(stderr='output.txt', mode='a'):
    <code>
  http://code.activestate.com/recipes/577564-context-manager-for-low-level-redirection-of-stdou/
  >>>

  """

    def __init__(self, stdout=os.devnull, stderr=os.devnull, mode='wb'):
        self.outfiles = (stdout, stderr)
        self.combine = stdout == stderr
        self.mode = mode
        self.saved_streams = None
        self.fds = None
        self.saved_fds = None
        self.null_fds = None
        self.null_streams = None

    def __enter__(self):
        self.saved_streams = saved_streams = (sys.__stdout__, sys.__stderr__)
        self.fds = fds = [s.fileno() for s in saved_streams]
        self.saved_fds = [os.dup(fd) for fd in fds]
        for s in saved_streams:
            s.flush()
        if self.combine:
            null_streams = [open(self.outfiles[0], self.mode, 0)] * 2
            if self.outfiles[0] != os.devnull:
                sys.stdout, sys.stderr = [os.fdopen(fd, 'wb', 0) for fd in fds]
        else:
            null_streams = [open(f, self.mode, 0) for f in self.outfiles]
        self.null_fds = null_fds = [s.fileno() for s in null_streams]
        self.null_streams = null_streams
        for null_fd, fd in zip(null_fds, fds):
            os.dup2(null_fd, fd)

    def __exit__(self, *args):
        for s in self.saved_streams:
            s.flush()
        for saved_fd, fd in zip(self.saved_fds, self.fds):
            os.dup2(saved_fd, fd)
        sys.stdout, sys.stderr = self.saved_streams
        for s in self.null_streams:
            s.close()
        for fd in self.saved_fds:
            os.close(fd)
        return False