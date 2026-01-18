import os
import subprocess
from .errors import HookError
class CommitMsgShellHook(ShellHook):
    """commit-msg shell hook."""

    def __init__(self, controldir) -> None:
        filepath = os.path.join(controldir, 'hooks', 'commit-msg')

        def prepare_msg(*args):
            import tempfile
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as f:
                f.write(args[0])
            return (path,)

        def clean_msg(success, *args):
            if success:
                with open(args[0], 'rb') as f:
                    new_msg = f.read()
                os.unlink(args[0])
                return new_msg
            os.unlink(args[0])
        ShellHook.__init__(self, 'commit-msg', filepath, 1, prepare_msg, clean_msg, controldir)