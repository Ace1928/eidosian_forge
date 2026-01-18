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
def _build_argument_spec(self, additional_arg_spec, validate_etag_support):
    merged_arg_spec = dict()
    merged_arg_spec.update(self.ONEVIEW_COMMON_ARGS)
    if validate_etag_support:
        merged_arg_spec.update(self.ONEVIEW_VALIDATE_ETAG_ARGS)
    if additional_arg_spec:
        merged_arg_spec.update(additional_arg_spec)
    return merged_arg_spec