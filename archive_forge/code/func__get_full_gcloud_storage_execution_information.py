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
def _get_full_gcloud_storage_execution_information(self, args):
    top_level_flags, env_variables = self._translate_top_level_flags()
    header_flags = self._translate_headers()
    flags_from_boto, env_vars_from_boto = self._translate_boto_config()
    env_variables.update(env_vars_from_boto)
    gcloud_binary_path = _get_validated_gcloud_binary_path()
    gcloud_storage_command = [gcloud_binary_path] + args + top_level_flags + header_flags + flags_from_boto
    return (env_variables, gcloud_storage_command)