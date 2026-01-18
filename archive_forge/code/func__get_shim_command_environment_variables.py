from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
def _get_shim_command_environment_variables(self):
    subprocess_envs = os.environ.copy()
    subprocess_envs.update(self._translated_env_variables)
    return subprocess_envs