from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def _get_cli_class(self, cli_class=None):
    if cli_class is not None:
        return cli_class(self.subdomain, self.domain, self.username, self.secret_key, self.master_password, self.service_account_token)
    version = OnePassCLIBase.get_current_version()
    for cls in OnePassCLIBase.__subclasses__():
        if cls.supports_version == version.split('.')[0]:
            try:
                return cls(self.subdomain, self.domain, self.username, self.secret_key, self.master_password, self.service_account_token, self.account_id, self.connect_host, self.connect_token)
            except TypeError as e:
                raise AnsibleLookupError(e)
    raise AnsibleLookupError('op version %s is unsupported' % version)