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
def GetConfigStore(config_name=None):
    """Gets the config sqlite store for a given config name.

  Args:
    config_name: string, The configuration name to get the config store for.

  Returns:
    SqliteConfigStore, The corresponding config store, or None if no config.
  """
    if config_name is None:
        try:
            config_name = named_configs.ConfigurationStore.ActiveConfig().name
        except named_configs.NamedConfigFileAccessError:
            return None
    return _GetSqliteStore(config_name)