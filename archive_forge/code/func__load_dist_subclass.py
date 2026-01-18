from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _load_dist_subclass(cls, *args, **kwargs):
    """
    Used for derivative implementations
    """
    subclass = None
    distro = kwargs['module'].params['distro']
    if distro is not None:
        for sc in cls.__subclasses__():
            if sc.distro is not None and sc.distro == distro:
                subclass = sc
    if subclass is None:
        subclass = cls
    return super(cls, subclass).__new__(subclass)