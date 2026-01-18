from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class PgHbaRuleValueError(PgHbaRuleError):
    """
    This exception is raised when a new parsed rule is a changed version of an existing rule.
    """