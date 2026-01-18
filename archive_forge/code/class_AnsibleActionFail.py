from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
class AnsibleActionFail(AnsibleAction):
    """ an action runtime failure"""

    def __init__(self, message='', obj=None, show_content=True, suppress_extended_error=False, orig_exc=None, result=None):
        super(AnsibleActionFail, self).__init__(message=message, obj=obj, show_content=show_content, suppress_extended_error=suppress_extended_error, orig_exc=orig_exc, result=result)
        self.result.update({'failed': True, 'msg': message, 'exception': traceback.format_exc()})