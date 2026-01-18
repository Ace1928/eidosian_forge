from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def _git_repo_info(repo_path):
    """ returns a string containing git branch, commit id and commit date """
    result = None
    if os.path.exists(repo_path):
        if os.path.isfile(repo_path):
            try:
                with open(repo_path) as f:
                    gitdir = yaml_load(f).get('gitdir')
                if os.path.isabs(gitdir):
                    repo_path = gitdir
                else:
                    repo_path = os.path.join(repo_path[:-4], gitdir)
            except (IOError, AttributeError):
                return ''
        with open(os.path.join(repo_path, 'HEAD')) as f:
            line = f.readline().rstrip('\n')
            if line.startswith('ref:'):
                branch_path = os.path.join(repo_path, line[5:])
            else:
                branch_path = None
        if branch_path and os.path.exists(branch_path):
            branch = '/'.join(line.split('/')[2:])
            with open(branch_path) as f:
                commit = f.readline()[:10]
        else:
            commit = line[:10]
            branch = 'detached HEAD'
            branch_path = os.path.join(repo_path, 'HEAD')
        date = time.localtime(os.stat(branch_path).st_mtime)
        if time.daylight == 0:
            offset = time.timezone
        else:
            offset = time.altzone
        result = '({0} {1}) last updated {2} (GMT {3:+04d})'.format(branch, commit, time.strftime('%Y/%m/%d %H:%M:%S', date), int(offset / -36))
    else:
        result = ''
    return result