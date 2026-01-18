import os
import subprocess
from .errors import HookError
class ShellHook(Hook):
    """Hook by executable file.

    Implements standard githooks(5) [0]:

    [0] http://www.kernel.org/pub/software/scm/git/docs/githooks.html
    """

    def __init__(self, name, path, numparam, pre_exec_callback=None, post_exec_callback=None, cwd=None) -> None:
        """Setup shell hook definition.

        Args:
          name: name of hook for error messages
          path: absolute path to executable file
          numparam: number of requirements parameters
          pre_exec_callback: closure for setup before execution
            Defaults to None. Takes in the variable argument list from the
            execute functions and returns a modified argument list for the
            shell hook.
          post_exec_callback: closure for cleanup after execution
            Defaults to None. Takes in a boolean for hook success and the
            modified argument list and returns the final hook return value
            if applicable
          cwd: working directory to switch to when executing the hook
        """
        self.name = name
        self.filepath = path
        self.numparam = numparam
        self.pre_exec_callback = pre_exec_callback
        self.post_exec_callback = post_exec_callback
        self.cwd = cwd

    def execute(self, *args):
        """Execute the hook with given args."""
        if len(args) != self.numparam:
            raise HookError('Hook %s executed with wrong number of args.                             Expected %d. Saw %d. args: %s' % (self.name, self.numparam, len(args), args))
        if self.pre_exec_callback is not None:
            args = self.pre_exec_callback(*args)
        try:
            ret = subprocess.call([os.path.relpath(self.filepath, self.cwd), *list(args)], cwd=self.cwd)
            if ret != 0:
                if self.post_exec_callback is not None:
                    self.post_exec_callback(0, *args)
                raise HookError('Hook %s exited with non-zero status %d' % (self.name, ret))
            if self.post_exec_callback is not None:
                return self.post_exec_callback(1, *args)
        except OSError:
            if self.post_exec_callback is not None:
                self.post_exec_callback(0, *args)