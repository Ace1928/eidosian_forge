from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
@Cache
def GetFeatureFlagsConfig(account_id, project_id):
    """Gets the feature flags config.

  If the feature flags config file does not exist or is stale, download and save
  the feature flags config. Otherwise, read the feature flags config. Errors
  will be logged, but will not interrupt normal operation.

  Args:
    account_id: str, account ID.
    project_id: str, project ID


  Returns:
    A FeatureFlagConfig, or None.
  """
    feature_flags_config_path = config.Paths().feature_flags_config_path
    with _FEATURE_FLAGS_LOCK:
        yaml_data = None
        if IsFeatureFlagsConfigStale(feature_flags_config_path):
            yaml_data = FetchFeatureFlagsConfig()
            try:
                file_utils.WriteFileContents(feature_flags_config_path, yaml_data or '')
            except file_utils.Error as e:
                logging.warning('Unable to write feature flags config [%s]: %s. Please ensure that this path is writeable.', feature_flags_config_path, e)
        else:
            try:
                yaml_data = file_utils.ReadFileContents(feature_flags_config_path)
            except file_utils.Error as e:
                logging.warning('Unable to read feature flags config [%s]: %s. Please ensure that this path is readable.', feature_flags_config_path, e)
    if yaml_data:
        return FeatureFlagsConfig(yaml_data, account_id, project_id)
    return None