from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection

        :param filter_existing: tuple consisting of the function at (index 0) and the args at (index 1)
        CAUTION: the function should always take in self.existing in its first parameter
        :param kwargs: kwargs to be passed to ansible module exit_json()
        filter_existing is not passed via kwargs since it cant handle function type and should not be exposed to user
        