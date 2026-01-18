from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
def _CreateDefaultConfig(force_create):
    """Create the default configuration and migrate legacy properties.

  This will only do anything if there are no existing configurations.  If that
  is true, it will create one called default.  If there are existing legacy
  properties, it will populate the new configuration with those settings.
  The old file will be marked as deprecated.

  Args:
    force_create: bool, If False and no legacy properties exist to be migrated
      this will not physically create the default configuration.  This is ok
      as long as we are strictly reading properties from this configuration.

  Returns:
    str, The default configuration name.
  """
    paths = config.Paths()
    try:
        if not os.path.exists(paths.named_config_activator_path):
            legacy_properties = _GetAndDeprecateLegacyProperties(paths)
            if legacy_properties or force_create:
                file_utils.MakeDir(paths.named_config_directory)
                target_file = _FileForConfig(DEFAULT_CONFIG_NAME, paths)
                file_utils.WriteFileContents(target_file, legacy_properties)
                file_utils.WriteFileContents(paths.named_config_activator_path, DEFAULT_CONFIG_NAME)
    except file_utils.Error as e:
        raise NamedConfigFileAccessError('Failed to create the default configuration. Ensure your have the correct permissions on: [{0}]'.format(paths.named_config_directory), e)
    return DEFAULT_CONFIG_NAME