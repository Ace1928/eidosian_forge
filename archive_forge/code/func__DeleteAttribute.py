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
def _DeleteAttribute(self, config_attr: str):
    """Deletes a specified attribute from the config."""
    try:
        self._Execute('DELETE FROM config WHERE config_attr = ?', (config_attr,))
        with self._cursor as cur:
            if cur.RowCount() < 1:
                logging.warning('Could not delete attribute [%s] from cache in config store [%s].', config_attr, self._config_name)
    except sqlite3.OperationalError as e:
        logging.warning('Could not delete attribute [%s] from cache: %s', config_attr, str(e))