from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class Ocfs2(Filesystem):
    MKFS = 'mkfs.ocfs2'
    MKFS_FORCE_FLAGS = ['-Fx']