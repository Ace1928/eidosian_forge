import os, tempfile
def _rl_getuid():
    if hasattr(os, 'getuid'):
        return os.getuid()
    else:
        return ''