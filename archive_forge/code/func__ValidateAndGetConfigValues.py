from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
def _ValidateAndGetConfigValues(self):
    """Parses the user's config file to aggregate non-PII config values.

    Returns:
      A comma-delimited string of config values explicitly set by the user in
      key:value pairs, sorted alphabetically by key.
    """
    config_values = []
    invalid_value_string = 'INVALID'

    def GetAndValidateConfigValue(section, category, validation_fn):
        try:
            config_value = boto.config.get_value(section, category)
            if config_value and validation_fn(config_value):
                config_values.append((category, config_value))
            elif config_value:
                config_values.append((category, invalid_value_string))
        except:
            config_values.append((category, invalid_value_string))
    for section, bool_category in (('Boto', 'https_validate_certificates'), ('GSUtil', 'disable_analytics_prompt'), ('GSUtil', 'use_magicfile'), ('GSUtil', 'tab_completion_time_logs')):
        GetAndValidateConfigValue(section=section, category=bool_category, validation_fn=lambda val: str(val).lower() in ('true', 'false'))
    small_int_threshold = 2000
    for section, small_int_category in (('Boto', 'debug'), ('Boto', 'http_socket_timeout'), ('Boto', 'num_retries'), ('Boto', 'max_retry_delay'), ('GSUtil', 'default_api_version'), ('GSUtil', 'sliced_object_download_max_components'), ('GSUtil', 'parallel_process_count'), ('GSUtil', 'parallel_thread_count'), ('GSUtil', 'software_update_check_period'), ('GSUtil', 'tab_completion_timeout'), ('OAuth2', 'oauth2_refresh_retries')):
        GetAndValidateConfigValue(section=section, category=small_int_category, validation_fn=lambda val: str(val).isdigit() and int(val) < small_int_threshold)
    for section, large_int_category in (('GSUtil', 'resumable_threshold'), ('GSUtil', 'rsync_buffer_lines'), ('GSUtil', 'task_estimation_threshold')):
        GetAndValidateConfigValue(section=section, category=large_int_category, validation_fn=lambda val: str(val).isdigit())
    for section, data_size_category in (('GSUtil', 'parallel_composite_upload_component_size'), ('GSUtil', 'parallel_composite_upload_threshold'), ('GSUtil', 'sliced_object_download_component_size'), ('GSUtil', 'sliced_object_download_threshold')):
        config_value = boto.config.get_value(section, data_size_category)
        if config_value:
            try:
                size_in_bytes = HumanReadableToBytes(config_value)
                config_values.append((data_size_category, size_in_bytes))
            except ValueError:
                config_values.append((data_size_category, invalid_value_string))
    GetAndValidateConfigValue(section='GSUtil', category='check_hashes', validation_fn=lambda val: val in ('if_fast_else_fail', 'if_fast_else_skip', 'always', 'never'))
    GetAndValidateConfigValue(section='GSUtil', category='content_language', validation_fn=lambda val: val.isalpha() and len(val) <= 3)
    GetAndValidateConfigValue(section='GSUtil', category='json_api_version', validation_fn=lambda val: val[0].lower() == 'v' and val[1:].isdigit())
    GetAndValidateConfigValue(section='GSUtil', category='prefer_api', validation_fn=lambda val: val in ('json', 'xml'))
    GetAndValidateConfigValue(section='OAuth2', category='token_cache', validation_fn=lambda val: val in ('file_system', 'in_memory'))
    return ','.join(sorted(['{0}:{1}'.format(config[0], config[1]) for config in config_values]))