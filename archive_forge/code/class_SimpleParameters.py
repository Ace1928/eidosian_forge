from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class SimpleParameters(Parameters):
    api_attributes = ['strategy']
    updatables = ['strategy', 'rules']
    returnables = ['strategy', 'rules']