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
def _ActiveConfigNameFromFile():
    """Gets the name of the user's active named config according to the file.

  Returns:
    str, The name of the active configuration or None.
  """
    path = config.Paths().named_config_activator_path
    is_invalid = False
    try:
        config_name = file_utils.ReadFileContents(path)
        if config_name:
            if _IsValidConfigName(config_name, allow_reserved=True):
                return config_name
            else:
                is_invalid = True
    except file_utils.MissingFileError:
        pass
    except file_utils.Error as exc:
        raise NamedConfigFileAccessError('Active configuration name could not be read from: [{0}]. Ensure you have sufficient read permissions on required active configuration in [{1}]'.format(path, config.Paths().named_config_directory), exc)
    if is_invalid:
        os.remove(path)
    return None