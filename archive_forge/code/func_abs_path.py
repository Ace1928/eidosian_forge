import base64
import os.path
import uuid
from .. import __version__
def abs_path(path):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        cwd = os.environ.get('PWD') or os.getcwd()
        path = os.path.join(cwd, path)
    return path