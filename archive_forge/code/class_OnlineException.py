from __future__ import (absolute_import, division, print_function)
import json
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
class OnlineException(Exception):

    def __init__(self, message):
        self.message = message