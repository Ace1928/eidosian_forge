from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def _enablerepos_with_error_checking(self):
    if len(self.enablerepo) == 1:
        try:
            self.yum_base.repos.enableRepo(self.enablerepo[0])
        except yum.Errors.YumBaseError as e:
            if u'repository not found' in to_text(e):
                self.module.fail_json(msg='Repository %s not found.' % self.enablerepo[0])
            else:
                raise e
    else:
        for rid in self.enablerepo:
            try:
                self.yum_base.repos.enableRepo(rid)
            except yum.Errors.YumBaseError as e:
                if u'repository not found' in to_text(e):
                    self.module.warn('Repository %s not found.' % rid)
                else:
                    raise e