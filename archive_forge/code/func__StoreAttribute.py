from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
def _StoreAttribute(self, config_attr: str, config_value):
    """Stores the input config attributes to the record of config_name in the cache.

    Args:
      config_attr: string, the primary key of the attribute to store.
      config_value: obj, the value of the config key attribute.
    """
    self._Execute('REPLACE INTO config (config_attr, value) VALUES (?,?)', (config_attr, config_value))