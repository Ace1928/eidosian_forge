from __future__ import (annotations, absolute_import, division, print_function)
import base64
import logging
import os
import re
import traceback
import json
import tempfile
import shlex
import subprocess
import time
import typing as t
import xml.etree.ElementTree as ET
from inspect import getfullargspec
from urllib.parse import urlunsplit
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.plugins.shell.powershell import ShellBase as PowerShellBase
from ansible.utils.hashing import secure_hash
from ansible.utils.display import Display
def _kerb_auth(self, principal: str, password: str) -> None:
    if password is None:
        password = ''
    self._kerb_ccache = tempfile.NamedTemporaryFile()
    display.vvvvv('creating Kerberos CC at %s' % self._kerb_ccache.name)
    krb5ccname = 'FILE:%s' % self._kerb_ccache.name
    os.environ['KRB5CCNAME'] = krb5ccname
    krb5env = dict(PATH=os.environ['PATH'], KRB5CCNAME=krb5ccname)
    kinit_env_vars = self.get_option('kinit_env_vars')
    for var in kinit_env_vars:
        if var not in krb5env and var in os.environ:
            krb5env[var] = os.environ[var]
    kinit_cmdline = [self._kinit_cmd]
    kinit_args = self.get_option('kinit_args')
    if kinit_args:
        kinit_args = [to_text(a) for a in shlex.split(kinit_args) if a.strip()]
        kinit_cmdline.extend(kinit_args)
    elif boolean(self.get_option('_extras').get('ansible_winrm_kerberos_delegation', False)):
        kinit_cmdline.append('-f')
    kinit_cmdline.append(principal)
    if HAS_PEXPECT:
        proc_mechanism = 'pexpect'
        command = kinit_cmdline.pop(0)
        password = to_text(password, encoding='utf-8', errors='surrogate_or_strict')
        display.vvvv('calling kinit with pexpect for principal %s' % principal)
        try:
            child = pexpect.spawn(command, kinit_cmdline, timeout=60, env=krb5env, echo=False)
        except pexpect.ExceptionPexpect as err:
            err_msg = "Kerberos auth failure when calling kinit cmd '%s': %s" % (command, to_native(err))
            raise AnsibleConnectionFailure(err_msg)
        try:
            child.expect('.*:')
            child.sendline(password)
        except OSError as err:
            display.vvvv('kinit with pexpect raised OSError: %s' % to_native(err))
        stderr = child.read()
        child.wait()
        rc = child.exitstatus
    else:
        proc_mechanism = 'subprocess'
        b_password = to_bytes(password, encoding='utf-8', errors='surrogate_or_strict')
        display.vvvv('calling kinit with subprocess for principal %s' % principal)
        try:
            p = subprocess.Popen(kinit_cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=krb5env)
        except OSError as err:
            err_msg = "Kerberos auth failure when calling kinit cmd '%s': %s" % (self._kinit_cmd, to_native(err))
            raise AnsibleConnectionFailure(err_msg)
        stdout, stderr = p.communicate(b_password + b'\n')
        rc = p.returncode != 0
    if rc != 0:
        exp_msg = to_native(stderr.strip())
        exp_msg = exp_msg.replace(to_native(password), '<redacted>')
        err_msg = 'Kerberos auth failure for principal %s with %s: %s' % (principal, proc_mechanism, exp_msg)
        raise AnsibleConnectionFailure(err_msg)
    display.vvvvv('kinit succeeded for principal %s' % principal)