from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
class ParametersMixinBase(object):
    """
    Base Class for the Parameters Mixins.

    Provides a function to verify no duplicate parameters are set.
    """

    def validate_parameters(self):
        parameters = self.foreman_params.get('parameters')
        if parameters is not None:
            parameter_names = [param['name'] for param in parameters]
            duplicate_params = set([x for x in parameter_names if parameter_names.count(x) > 1])
            if duplicate_params:
                self.fail_json(msg="There are duplicate keys in 'parameters': {0}.".format(duplicate_params))