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
def _validate_credential_source(self, parent_profile, credential_source):
    if self._credential_sourcer is None:
        raise InvalidConfigError(error_msg=f'The credential_source "{credential_source}" is specified in profile "{parent_profile}", but no source provider was configured.')
    if not self._credential_sourcer.is_supported(credential_source):
        raise InvalidConfigError(error_msg=f'The credential source "{credential_source}" referenced in profile "{parent_profile}" is not valid.')