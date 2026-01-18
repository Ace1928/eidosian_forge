from __future__ import annotations
import ast
import datetime
import os
import re
import sys
from io import BytesIO, TextIOWrapper
import yaml
import yaml.reader
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.yaml import SafeLoader
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.loader import AnsibleLoader
class NoArgsAnsibleModule(AnsibleModule):
    """AnsibleModule that does not actually load params. This is used to get access to the
    methods within AnsibleModule without having to fake a bunch of data
    """

    def _load_params(self):
        self.params = {'_ansible_selinux_special_fs': [], '_ansible_remote_tmp': '/tmp', '_ansible_keep_remote_files': False, '_ansible_check_mode': False}