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
def _create_client(self):
    """Create an STS client using the source credentials."""
    frozen_credentials = self._source_credentials.get_frozen_credentials()
    return self._client_creator('sts', aws_access_key_id=frozen_credentials.access_key, aws_secret_access_key=frozen_credentials.secret_key, aws_session_token=frozen_credentials.token)