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
class OneViewModuleException(Exception):
    """
    OneView base Exception.

    Attributes:
       msg (str): Exception message.
       oneview_response (dict): OneView rest response.
   """

    def __init__(self, data):
        self.msg = None
        self.oneview_response = None
        if isinstance(data, six.string_types):
            self.msg = data
        else:
            self.oneview_response = data
            if data and isinstance(data, dict):
                self.msg = data.get('message')
        if self.oneview_response:
            Exception.__init__(self, self.msg, self.oneview_response)
        else:
            Exception.__init__(self, self.msg)