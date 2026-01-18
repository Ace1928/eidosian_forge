import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
def _build_mapping(self, mapping):
    var_mapping = {}
    if mapping is None:
        var_mapping['access_key'] = self.ACCESS_KEY
        var_mapping['secret_key'] = self.SECRET_KEY
        var_mapping['token'] = self.TOKENS
        var_mapping['expiry_time'] = self.EXPIRY_TIME
    else:
        var_mapping['access_key'] = mapping.get('access_key', self.ACCESS_KEY)
        var_mapping['secret_key'] = mapping.get('secret_key', self.SECRET_KEY)
        var_mapping['token'] = mapping.get('token', self.TOKENS)
        if not isinstance(var_mapping['token'], list):
            var_mapping['token'] = [var_mapping['token']]
        var_mapping['expiry_time'] = mapping.get('expiry_time', self.EXPIRY_TIME)
    return var_mapping