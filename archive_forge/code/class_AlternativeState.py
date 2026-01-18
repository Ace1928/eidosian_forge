from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
class AlternativeState:
    PRESENT = 'present'
    SELECTED = 'selected'
    ABSENT = 'absent'
    AUTO = 'auto'

    @classmethod
    def to_list(cls):
        return [cls.PRESENT, cls.SELECTED, cls.ABSENT, cls.AUTO]