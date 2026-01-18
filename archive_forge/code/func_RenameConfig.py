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
@staticmethod
def RenameConfig(config_name, new_name):
    """Renames an existing named configuration.

    Args:
      config_name: str, The name of the configuration to rename.
      new_name: str, The new name of the configuration.

    Raises:
      NamedConfigError: If the configuration does not exist, or if the
        configuration with new_name exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
    _EnsureValidConfigName(new_name, allow_reserved=True)
    paths = config.Paths()
    file_path = _FileForConfig(config_name, paths)
    if file_path and (not os.path.exists(file_path)):
        raise NamedConfigError('Cannot rename configuration [{0}], it does not exist.'.format(config_name))
    if config_name == _EffectiveActiveConfigName():
        raise NamedConfigError('Cannot rename configuration [{0}], it is the currently active configuration.'.format(config_name))
    if config_name == _ActiveConfigNameFromFile():
        raise NamedConfigError('Cannot rename configuration [{0}], it is currently set as the active configuration in your gcloud properties.'.format(config_name))
    new_file_path = _FileForConfig(new_name, paths)
    if new_file_path and os.path.exists(new_file_path):
        raise NamedConfigError('Cannot rename configuration [{0}], [{1}] already exists.'.format(config_name, new_name))
    try:
        contents = file_utils.ReadFileContents(file_path)
        file_utils.WriteFileContents(new_file_path, contents)
        os.remove(file_path)
    except file_utils.Error as e:
        raise NamedConfigFileAccessError('Failed to rename configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_activator_path), e)