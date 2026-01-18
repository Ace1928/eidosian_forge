from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class AsmSignatureSetsFactParameters(BaseParameters):
    api_map = {'isUserDefined': 'is_user_defined', 'assignToPolicyByDefault': 'assign_to_policy_by_default', 'defaultAlarm': 'default_alarm', 'defaultBlock': 'default_block', 'defaultLearn': 'default_learn'}
    returnables = ['name', 'id', 'type', 'category', 'is_user_defined', 'assign_to_policy_by_default', 'default_alarm', 'default_block', 'default_learn']

    @property
    def is_user_defined(self):
        return flatten_boolean(self._values['is_user_defined'])

    @property
    def assign_to_policy_by_default(self):
        return flatten_boolean(self._values['assign_to_policy_by_default'])

    @property
    def default_alarm(self):
        return flatten_boolean(self._values['default_alarm'])

    @property
    def default_block(self):
        return flatten_boolean(self._values['default_block'])

    @property
    def default_learn(self):
        return flatten_boolean(self._values['default_learn'])