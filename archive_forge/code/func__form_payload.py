from __future__ import (absolute_import, division, print_function)
import base64
import os
import json
from stat import S_IRUSR, S_IWUSR
from ansible import constants as C
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
def _form_payload(self):
    return 'grant_type=refresh_token&client_id=%s&refresh_token=%s' % (self.client_id, self.access_token)