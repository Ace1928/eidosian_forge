from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def __get_auth_dict():
    return dict(type='dict', apply_defaults=True, required=True, required_one_of=[['hostname', 'url']], options=dict(url=dict(type='str', fallback=(env_fallback, ['OVIRT_URL'])), hostname=dict(type='str', fallback=(env_fallback, ['OVIRT_HOSTNAME'])), username=dict(type='str', fallback=(env_fallback, ['OVIRT_USERNAME'])), password=dict(type='str', fallback=(env_fallback, ['OVIRT_PASSWORD']), no_log=True), insecure=dict(type='bool', default=False), token=dict(type='str', fallback=(env_fallback, ['OVIRT_TOKEN']), no_log=False), ca_file=dict(type='str', fallback=(env_fallback, ['OVIRT_CAFILE'])), compress=dict(type='bool', default=True), timeout=dict(type='int', default=0), kerberos=dict(type='bool'), headers=dict(type='dict')))