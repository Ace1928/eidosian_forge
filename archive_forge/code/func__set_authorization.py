from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def _set_authorization(self):
    if version.parse(ansible_version) < version.parse('2.11'):
        token = self.get_option('token')
    else:
        self.templar.available_variables = self._vars
        token = self.templar.template(self.get_option('token'), fail_on_undefined=False)
    if token:
        if isinstance(token, dict):
            self.headers.update({'Authorization': f'{token['type'].capitalize()} {token['value']}'})
        else:
            self.headers.update({'Authorization': 'Token %s' % token})