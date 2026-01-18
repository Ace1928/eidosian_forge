import os
import re
import subprocess  # nosec
from hacking import core
def _get_commit_title(self):
    try:
        subp = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gitdir = subp.communicate()[0].rstrip()
    except OSError:
        return None
    if not os.path.exists(gitdir):
        return None
    subp = subprocess.Popen(['git', 'log', '--no-merges', '--pretty=%s', '-1'], stdout=subprocess.PIPE)
    title = subp.communicate()[0]
    if subp.returncode:
        raise Exception('git log failed with code %s' % subp.returncode)
    return title.decode('utf-8')