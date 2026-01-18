from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
import six
from six.moves import configparser
def __Load(self, properties_path):
    """Loads properties from the given file.

    Overwrites anything already known.

    Args:
      properties_path: str, Path to the file containing properties info.
    """
    parsed_config = configparser.ConfigParser()
    try:
        parsed_config.read(properties_path)
    except configparser.ParsingError as e:
        raise PropertiesParseError(str(e))
    for section in parsed_config.sections():
        if section not in self._properties:
            self._properties[section] = {}
        self._properties[section].update(dict(parsed_config.items(section)))