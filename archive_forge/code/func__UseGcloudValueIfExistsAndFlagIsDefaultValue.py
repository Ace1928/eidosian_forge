import codecs
import copy
import http.client
import json
import logging
import os
import pkgutil
import platform
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, TextIO
from absl import app
from absl import flags
from google.auth import version as google_auth_version
from google.oauth2 import credentials as google_oauth2
import googleapiclient
import httplib2
import oauth2client_4_0.client
import requests
import urllib3
from utils import bq_error
from utils import bq_logging
from pyglib import stringutil
def _UseGcloudValueIfExistsAndFlagIsDefaultValue(flag_values, flag_name: str, gcloud_config_section: Dict[str, str], gcloud_property_name: str):
    """Updates flag if it's using the default and the gcloud value exists."""
    if not gcloud_config_section:
        return
    if gcloud_property_name not in gcloud_config_section:
        return
    flag = flag_values[flag_name]
    gcloud_value = gcloud_config_section[gcloud_property_name]
    logging.debug('Gcloud config exists for %s', gcloud_property_name)
    if flag.using_default_value:
        logging.info('The `%s` flag is using a default value and a value is set in gcloud, using that: %s', flag_name, gcloud_value)
        UpdateFlag(flag_values, flag_name, gcloud_value)
    elif flag.value != gcloud_value:
        logging.warning('Executing with different configuration than in gcloud.The flag "%s" has become set to "%s" but gcloud sets "%s" as "%s".To update the gcloud value, start from `gcloud config list`.', flag_name, flag.value, gcloud_property_name, gcloud_value)