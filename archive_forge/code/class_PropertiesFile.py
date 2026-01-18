from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
import six
from six.moves import configparser
class PropertiesFile(object):
    """A class for loading and parsing property files."""

    def __init__(self, paths):
        """Creates a new PropertiesFile and load from the given paths.

    Args:
      paths: [str], List of files to load properties from, in order.
    """
        self._properties = {}
        for properties_path in paths:
            if properties_path:
                self.__Load(properties_path)

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

    def Get(self, section, name):
        """Gets the value of the given property.

    Args:
      section: str, The section name of the property to get.
      name: str, The name of the property to get.

    Returns:
      str, The value for the given section and property, or None if it is not
        set.
    """
        try:
            return self._properties[section][name]
        except KeyError:
            return None

    def AllProperties(self):
        """Returns a dictionary of properties in the file."""
        return dict(self._properties)