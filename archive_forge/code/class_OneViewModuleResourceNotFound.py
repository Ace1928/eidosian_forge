from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
class OneViewModuleResourceNotFound(OneViewModuleException):
    """
    OneView Resource Not Found Exception.
    The exception is raised when an associated resource was not found.

    Attributes:
       msg (str): Exception message.
    """
    pass