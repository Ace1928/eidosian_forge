from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
class DragonFlyBsdGroup(FreeBsdGroup):
    """
    This is a DragonFlyBSD Group manipulation class.
    It inherits all behaviors from FreeBsdGroup class.
    """
    platform = 'DragonFly'