import os
import subprocess
from .errors import HookError
class PreCommitShellHook(ShellHook):
    """pre-commit shell hook."""

    def __init__(self, cwd, controldir) -> None:
        filepath = os.path.join(controldir, 'hooks', 'pre-commit')
        ShellHook.__init__(self, 'pre-commit', filepath, 0, cwd=cwd)