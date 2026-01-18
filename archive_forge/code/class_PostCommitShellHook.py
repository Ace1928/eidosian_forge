import os
import subprocess
from .errors import HookError
class PostCommitShellHook(ShellHook):
    """post-commit shell hook."""

    def __init__(self, controldir) -> None:
        filepath = os.path.join(controldir, 'hooks', 'post-commit')
        ShellHook.__init__(self, 'post-commit', filepath, 0, cwd=controldir)