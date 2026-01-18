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
def _get_package_info(module, package, python_bin=None):
    """This is only needed for special packages which do not show up in pip freeze

    pip and setuptools fall into this category.

    :returns: a string containing the version number if the package is
        installed.  None if the package is not installed.
    """
    if python_bin is None:
        return
    discovery_mechanism = 'pkg_resources'
    importlib_rc = module.run_command([python_bin, '-c', 'import importlib.metadata'])[0]
    if importlib_rc == 0:
        discovery_mechanism = 'importlib'
    rc, out, err = module.run_command([python_bin, '-c', _SPECIAL_PACKAGE_CHECKERS[discovery_mechanism][package]])
    if rc:
        formatted_dep = None
    else:
        formatted_dep = '%s==%s' % (package, out.strip())
    return formatted_dep