from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class PgHbaError(Exception):
    """
    This exception is raised when parsing the pg_hba file ends in an error.
    """