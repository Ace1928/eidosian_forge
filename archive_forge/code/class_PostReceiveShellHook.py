import os
import subprocess
from .errors import HookError
class PostReceiveShellHook(ShellHook):
    """post-receive shell hook."""

    def __init__(self, controldir) -> None:
        self.controldir = controldir
        filepath = os.path.join(controldir, 'hooks', 'post-receive')
        ShellHook.__init__(self, 'post-receive', path=filepath, numparam=0)

    def execute(self, client_refs):
        if not os.path.exists(self.filepath):
            return None
        try:
            env = os.environ.copy()
            env['GIT_DIR'] = self.controldir
            p = subprocess.Popen(self.filepath, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            in_data = b'\n'.join([b' '.join(ref) for ref in client_refs])
            out_data, err_data = p.communicate(in_data)
            if p.returncode != 0 or err_data:
                err_fmt = b'post-receive exit code: %d\n' + b'stdout:\n%s\nstderr:\n%s'
                err_msg = err_fmt % (p.returncode, out_data, err_data)
                raise HookError(err_msg.decode('utf-8', 'backslashreplace'))
            return out_data
        except OSError as err:
            raise HookError(repr(err)) from err