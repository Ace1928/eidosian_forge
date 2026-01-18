from __future__ import absolute_import, division, print_function
import abc
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
class DHParameterOpenSSL(DHParameterBase):

    def __init__(self, module):
        super(DHParameterOpenSSL, self).__init__(module)
        self.openssl_bin = module.get_bin_path('openssl', True)

    def _do_generate(self, module):
        """Actually generate the DH params."""
        fd, tmpsrc = tempfile.mkstemp()
        os.close(fd)
        module.add_cleanup_file(tmpsrc)
        command = [self.openssl_bin, 'dhparam', '-out', tmpsrc, str(self.size)]
        rc, dummy, err = module.run_command(command, check_rc=False)
        if rc != 0:
            raise DHParameterError(to_native(err))
        if self.backup:
            self.backup_file = module.backup_local(self.path)
        try:
            module.atomic_move(tmpsrc, self.path)
        except Exception as e:
            module.fail_json(msg='Failed to write to file %s: %s' % (self.path, str(e)))

    def _check_params_valid(self, module):
        """Check if the params are in the correct state"""
        command = [self.openssl_bin, 'dhparam', '-check', '-text', '-noout', '-in', self.path]
        rc, out, err = module.run_command(command, check_rc=False)
        result = to_native(out)
        if rc != 0:
            return False
        match = re.search('Parameters:\\s+\\((\\d+) bit\\).*', result)
        if not match:
            return False
        bits = int(match.group(1))
        if 'WARNING' in result or 'WARNING' in to_native(err):
            return False
        return bits == self.size