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
def _validate_supported_payload(self, resource, action, payload):
    """
        Check whether the payload only contains supported keys.
        Emits a warning for keys that are not part of the apidoc.

        :param resource: Plural name of the api resource to check
        :type resource: str
        :param action: Name of the action to check payload against
        :type action: str
        :param payload: API paylod to be checked
        :type payload: dict

        :return: The payload as it can be submitted to the API
        :rtype: dict
        """
    filtered_payload = self._resource_prepare_params(resource, action, payload)
    unsupported_parameters = set(payload.keys()) - set(_recursive_dict_keys(filtered_payload))
    if unsupported_parameters:
        warn_msg = 'The following parameters are not supported by your server when performing {0} on {1}: {2}. They were ignored.'
        self.warn(warn_msg.format(action, resource, unsupported_parameters))
    return filtered_payload