from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
class SecretVariables(object):

    @staticmethod
    def ensure_scaleway_secret_package(module):
        if not HAS_SCALEWAY_SECRET_PACKAGE:
            module.fail_json(msg=missing_required_lib('passlib[argon2]', url='https://passlib.readthedocs.io/en/stable/'), exception=SCALEWAY_SECRET_IMP_ERR)

    @staticmethod
    def dict_to_list(source_dict):
        return [dict(key=var[0], value=var[1]) for var in source_dict.items()]

    @staticmethod
    def list_to_dict(source_list, hashed=False):
        key_value = 'hashed_value' if hashed else 'value'
        return dict(((var['key'], var[key_value]) for var in source_list))

    @classmethod
    def decode(cls, secrets_list, values_list):
        secrets_dict = cls.list_to_dict(secrets_list, hashed=True)
        values_dict = cls.list_to_dict(values_list, hashed=False)
        for key in values_dict:
            if key in secrets_dict:
                if argon2.verify(values_dict[key], secrets_dict[key]):
                    secrets_dict[key] = values_dict[key]
                else:
                    secrets_dict[key] = secrets_dict[key]
        return cls.dict_to_list(secrets_dict)