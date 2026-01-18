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
def PersistProperty(self, section, name, value):
    """Persists a property to this configuration file.

    Args:
      section: str, The section name of the property to set.
      name: str, The name of the property to set.
      value: str, The value to set for the given property, or None to unset it.

    Raises:
      ReadOnlyConfigurationError: If you are trying to persist properties to
        the None configuration.
      NamedConfigError: If the configuration does not exist.
    """
    if not self.file_path:
        raise ReadOnlyConfigurationError(self.name)
    if not os.path.exists(self.file_path):
        raise NamedConfigError('Cannot set property in configuration [{0}], it does not exist.'.format(self.name))
    properties_file.PersistProperty(self.file_path, section, name, value)
    if self.is_active:
        ActivePropertiesFile.Invalidate(mark_changed=True)