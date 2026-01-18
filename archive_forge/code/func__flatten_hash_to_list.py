from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from ansible.errors import AnsibleFileNotFound
from ansible.plugins import AnsiblePlugin
from ansible.utils.display import Display
@staticmethod
def _flatten_hash_to_list(terms):
    ret = []
    for key in terms:
        ret.append({'key': key, 'value': terms[key]})
    return ret