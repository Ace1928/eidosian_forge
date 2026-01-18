from __future__ import absolute_import, division, print_function
import warnings
import datetime
import fnmatch
import locale as locale_module
import os
import random
import re
import shutil
import sys
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, string_types
from ansible.module_utils.urls import fetch_file
class PolicyRcD(object):
    """
    This class is a context manager for the /usr/sbin/policy-rc.d file.
    It allow the user to prevent dpkg to start the corresponding service when installing
    a package.
    https://people.debian.org/~hmh/invokerc.d-policyrc.d-specification.txt
    """

    def __init__(self, module):
        self.m = module
        if self.m.params['policy_rc_d'] is None:
            return
        if os.path.exists('/usr/sbin/policy-rc.d'):
            self.backup_dir = tempfile.mkdtemp(prefix='ansible')
        else:
            self.backup_dir = None

    def __enter__(self):
        """
        This method will be called when we enter the context, before we call `apt-get …`
        """
        if self.m.params['policy_rc_d'] is None:
            return
        if self.backup_dir:
            try:
                shutil.move('/usr/sbin/policy-rc.d', self.backup_dir)
            except Exception:
                self.m.fail_json(msg='Fail to move /usr/sbin/policy-rc.d to %s' % self.backup_dir)
        try:
            with open('/usr/sbin/policy-rc.d', 'w') as policy_rc_d:
                policy_rc_d.write('#!/bin/sh\nexit %d\n' % self.m.params['policy_rc_d'])
            os.chmod('/usr/sbin/policy-rc.d', 493)
        except Exception:
            self.m.fail_json(msg='Failed to create or chmod /usr/sbin/policy-rc.d')

    def __exit__(self, type, value, traceback):
        """
        This method will be called when we exit the context, after `apt-get …` is done
        """
        if self.m.params['policy_rc_d'] is None:
            return
        if self.backup_dir:
            try:
                shutil.move(os.path.join(self.backup_dir, 'policy-rc.d'), '/usr/sbin/policy-rc.d')
                os.rmdir(self.backup_dir)
            except Exception:
                self.m.fail_json(msg='Fail to move back %s to /usr/sbin/policy-rc.d' % os.path.join(self.backup_dir, 'policy-rc.d'))
        else:
            try:
                os.remove('/usr/sbin/policy-rc.d')
            except Exception:
                self.m.fail_json(msg='Fail to remove /usr/sbin/policy-rc.d (after package manipulation)')