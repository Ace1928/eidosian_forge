from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _set_md5sum(self):
    try:
        result = self.module.md5(self.want.fulldest)
        self.want.update({'md5sum': result})
    except ValueError:
        pass