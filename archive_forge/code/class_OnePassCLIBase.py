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
class OnePassCLIBase(with_metaclass(abc.ABCMeta, object)):
    bin = 'op'

    def __init__(self, subdomain=None, domain='1password.com', username=None, secret_key=None, master_password=None, service_account_token=None, account_id=None, connect_host=None, connect_token=None):
        self.subdomain = subdomain
        self.domain = domain
        self.username = username
        self.master_password = master_password
        self.secret_key = secret_key
        self.service_account_token = service_account_token
        self.account_id = account_id
        self.connect_host = connect_host
        self.connect_token = connect_token
        self._path = None
        self._version = None

    def _check_required_params(self, required_params):
        non_empty_attrs = dict(((param, getattr(self, param, None)) for param in required_params if getattr(self, param, None)))
        missing = set(required_params).difference(non_empty_attrs)
        if missing:
            prefix = 'Unable to sign in to 1Password. Missing required parameter'
            plural = ''
            suffix = ': {params}.'.format(params=', '.join(missing))
            if len(missing) > 1:
                plural = 's'
            msg = '{prefix}{plural}{suffix}'.format(prefix=prefix, plural=plural, suffix=suffix)
            raise AnsibleLookupError(msg)

    @abc.abstractmethod
    def _parse_field(self, data_json, field_name, section_title):
        """Main method for parsing data returned from the op command line tool"""

    def _run(self, args, expected_rc=0, command_input=None, ignore_errors=False, environment_update=None):
        command = [self.path] + args
        call_kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'stdin': subprocess.PIPE}
        if environment_update:
            env = os.environ.copy()
            env.update(environment_update)
            call_kwargs['env'] = env
        p = subprocess.Popen(command, **call_kwargs)
        out, err = p.communicate(input=command_input)
        rc = p.wait()
        if not ignore_errors and rc != expected_rc:
            raise AnsibleLookupError(to_text(err))
        return (rc, out, err)

    @abc.abstractmethod
    def assert_logged_in(self):
        """Check whether a login session exists"""

    @abc.abstractmethod
    def full_signin(self):
        """Performa full login"""

    @abc.abstractmethod
    def get_raw(self, item_id, vault=None, token=None):
        """Gets the specified item from the vault"""

    @abc.abstractmethod
    def signin(self):
        """Sign in using the master password"""

    @property
    def path(self):
        if self._path is None:
            self._path = get_bin_path(self.bin)
        return self._path

    @property
    def version(self):
        if self._version is None:
            self._version = self.get_current_version()
        return self._version

    @classmethod
    def get_current_version(cls):
        """Standalone method to get the op CLI version. Useful when determining which class to load
        based on the current version."""
        try:
            bin_path = get_bin_path(cls.bin)
        except ValueError:
            raise AnsibleLookupError("Unable to locate '%s' command line tool" % cls.bin)
        try:
            b_out = subprocess.check_output([bin_path, '--version'], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as cpe:
            raise AnsibleLookupError('Unable to get the op version: %s' % cpe)
        return to_text(b_out).strip()