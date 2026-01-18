import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecRecursiveMirror(self, source, dest):
    """Emulation of rm -rf out && cp -af in out."""
    if os.path.exists(dest):
        if os.path.isdir(dest):

            def _on_error(fn, path, excinfo):
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWRITE)
                fn(path)
            shutil.rmtree(dest, onerror=_on_error)
        else:
            if not os.access(dest, os.W_OK):
                os.chmod(dest, stat.S_IWRITE)
            os.unlink(dest)
    if os.path.isdir(source):
        shutil.copytree(source, dest)
    else:
        shutil.copy2(source, dest)