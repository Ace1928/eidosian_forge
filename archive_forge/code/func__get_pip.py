from __future__ import absolute_import, division, print_function
import argparse
import os
import re
import sys
import tempfile
import operator
import shlex
import traceback
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def _get_pip(module, env=None, executable=None):
    candidate_pip_basenames = ('pip2', 'pip')
    if PY3:
        candidate_pip_basenames = ('pip3',)
    pip = None
    if executable is not None:
        if os.path.isabs(executable):
            pip = executable
        else:
            candidate_pip_basenames = (executable,)
    elif executable is None and env is None and _have_pip_module():
        pip = [sys.executable, '-m', 'pip.__main__']
    if pip is None:
        if env is None:
            opt_dirs = []
            for basename in candidate_pip_basenames:
                pip = module.get_bin_path(basename, False, opt_dirs)
                if pip is not None:
                    break
            else:
                module.fail_json(msg='Unable to find any of %s to use.  pip needs to be installed.' % ', '.join(candidate_pip_basenames))
        else:
            venv_dir = os.path.join(env, 'bin')
            candidate_pip_basenames = (candidate_pip_basenames[0], 'pip')
            for basename in candidate_pip_basenames:
                candidate = os.path.join(venv_dir, basename)
                if os.path.exists(candidate) and is_executable(candidate):
                    pip = candidate
                    break
            else:
                module.fail_json(msg='Unable to find pip in the virtualenv, %s, ' % env + 'under any of these names: %s. ' % ', '.join(candidate_pip_basenames) + 'Make sure pip is present in the virtualenv.')
    if not isinstance(pip, list):
        pip = [pip]
    return pip