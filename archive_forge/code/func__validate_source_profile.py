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
def _validate_source_profile(self, parent_profile_name, source_profile_name):
    profiles = self._loaded_config.get('profiles', {})
    if source_profile_name not in profiles:
        raise InvalidConfigError(error_msg=f'The source_profile "{source_profile_name}" referenced in the profile "{parent_profile_name}" does not exist.')
    source_profile = profiles[source_profile_name]
    if source_profile_name not in self._visited_profiles:
        return
    if source_profile_name != parent_profile_name:
        raise InfiniteLoopConfigError(source_profile=source_profile_name, visited_profiles=self._visited_profiles)
    if not self._has_static_credentials(source_profile):
        raise InfiniteLoopConfigError(source_profile=source_profile_name, visited_profiles=self._visited_profiles)