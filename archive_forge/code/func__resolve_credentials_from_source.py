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
def _resolve_credentials_from_source(self, credential_source, profile_name):
    credentials = self._credential_sourcer.source_credentials(credential_source)
    if credentials is None:
        raise CredentialRetrievalError(provider=credential_source, error_msg='No credentials found in credential_source referenced in profile %s' % profile_name)
    return credentials