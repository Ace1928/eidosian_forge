import os
import sys
import subprocess
import gyp  # noqa: E402
def UnixifyPath(path):
    try:
        if not IsCygwin():
            return path
        out = subprocess.Popen(['cygpath', '-u', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = out.communicate()
        return stdout.decode('utf-8')
    except Exception:
        return path