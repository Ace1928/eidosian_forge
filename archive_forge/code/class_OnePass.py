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
class OnePass(object):

    def __init__(self, subdomain=None, domain='1password.com', username=None, secret_key=None, master_password=None, service_account_token=None, account_id=None, connect_host=None, connect_token=None, cli_class=None):
        self.subdomain = subdomain
        self.domain = domain
        self.username = username
        self.secret_key = secret_key
        self.master_password = master_password
        self.service_account_token = service_account_token
        self.account_id = account_id
        self.connect_host = connect_host
        self.connect_token = connect_token
        self.logged_in = False
        self.token = None
        self._config = OnePasswordConfig()
        self._cli = self._get_cli_class(cli_class)
        if (self.connect_host or self.connect_token) and None in (self.connect_host, self.connect_token):
            raise AnsibleOptionsError('connect_host and connect_token are required together')

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

    def set_token(self):
        if self._config.config_file_path and os.path.isfile(self._config.config_file_path):
            try:
                rc, out, err = self._cli.signin()
            except AnsibleLookupError as exc:
                test_strings = ('missing required parameters', 'unauthorized')
                if any((string in exc.message.lower() for string in test_strings)):
                    raise
                rc, out, err = self._cli.full_signin()
            self.token = out.strip()
        else:
            rc, out, err = self._cli.full_signin()
            self.token = out.strip()

    def assert_logged_in(self):
        logged_in = self._cli.assert_logged_in()
        if logged_in:
            self.logged_in = logged_in
            pass
        else:
            self.set_token()

    def get_raw(self, item_id, vault=None):
        rc, out, err = self._cli.get_raw(item_id, vault, self.token)
        return out

    def get_field(self, item_id, field, section=None, vault=None):
        output = self.get_raw(item_id, vault)
        if output:
            return self._cli._parse_field(output, field, section)
        return ''