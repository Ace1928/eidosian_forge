from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def _register_using_cli(self, username, password, token, auto_attach, activationkey, org_id, consumer_type, consumer_name, consumer_id, force_register, environment, release):
    """
            Register using the 'subscription-manager' command

            Raises:
              * Exception - if error occurs while running command
        """
    args = [SUBMAN_CMD, 'register']
    if force_register:
        args.extend(['--force'])
    if org_id:
        args.extend(['--org', org_id])
    if auto_attach:
        args.append('--auto-attach')
    if consumer_type:
        args.extend(['--type', consumer_type])
    if consumer_name:
        args.extend(['--name', consumer_name])
    if consumer_id:
        args.extend(['--consumerid', consumer_id])
    if environment:
        args.extend(['--environment', environment])
    if activationkey:
        args.extend(['--activationkey', activationkey])
    elif token:
        args.extend(['--token', token])
    else:
        if username:
            args.extend(['--username', username])
        if password:
            args.extend(['--password', password])
    if release:
        args.extend(['--release', release])
    rc, stderr, stdout = self.module.run_command(args, check_rc=True, expand_user_and_vars=False)