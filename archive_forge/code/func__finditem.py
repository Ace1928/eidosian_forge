from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def _finditem(obj, keys):
    """ if keys is a string, use it as a key
        if keys is a tuple, stop on the first valid key
        if no valid key is found, raise a KeyError """
    value = None
    if isinstance(keys, str):
        value = __finditem(obj, keys)
    elif isinstance(keys, tuple):
        for key in keys:
            value = __finditem(obj, key)
            if value is not None:
                break
    if value is not None:
        return value
    raise KeyError(str(keys))