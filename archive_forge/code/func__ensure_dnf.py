from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _ensure_dnf(self):
    locale = get_best_parsable_locale(self.module)
    os.environ['LC_ALL'] = os.environ['LC_MESSAGES'] = locale
    os.environ['LANGUAGE'] = os.environ['LANG'] = locale
    global dnf
    try:
        import dnf
        import dnf.cli
        import dnf.const
        import dnf.exceptions
        import dnf.package
        import dnf.subject
        import dnf.util
        HAS_DNF = True
    except ImportError:
        HAS_DNF = False
    if HAS_DNF:
        return
    system_interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python']
    if not has_respawned():
        interpreter = probe_interpreters_for_module(system_interpreters, 'dnf')
        if interpreter:
            respawn_module(interpreter)
    self.module.fail_json(msg='Could not import the dnf python module using {0} ({1}). Please install `python3-dnf` or `python2-dnf` package or ensure you have specified the correct ansible_python_interpreter. (attempted {2})'.format(sys.executable, sys.version.replace('\n', ''), system_interpreters), results=[])