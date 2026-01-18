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
def GetJSON(self, config_attr, required=False):
    """Gets the JSON value for this attribute.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      The JSON value for this attribute or None.
    """
    attr_value = self._LoadAttribute(config_attr, required)
    if attr_value is None:
        return None
    try:
        return json.loads(attr_value)
    except ValueError:
        return attr_value