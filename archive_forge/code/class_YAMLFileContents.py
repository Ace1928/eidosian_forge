import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
class YAMLFileContents(object):
    """Creates an argparse type that reads the contents of a YAML or JSON file.

  This is similar to argparse.FileType, but unlike FileType it does not leave
  a dangling file handle open. The argument stored in the argparse Namespace
  is the file's contents parsed as a YAML object.

  Attributes:
    validator: function, Function that will validate the provided input file
      contents.

  Returns:
    A function that accepts a filename that should be parsed as a YAML
    or JSON file.
  """

    def __init__(self, validator=None):
        if validator and (not callable(validator)):
            raise ArgumentTypeError('Validator must be callable')
        self.validator = validator

    def _AssertJsonLike(self, yaml_data):
        from googlecloudsdk.core import yaml
        if not (yaml.dict_like(yaml_data) or yaml.list_like(yaml_data)):
            raise ArgumentTypeError('Invalid YAML/JSON Data [{}]'.format(yaml_data))

    def _LoadSingleYamlDocument(self, name):
        """Returns the yaml data for a file or from stdin for a single document.

    YAML allows multiple documents in a single file by using `---` as a
    separator between documents. See https://yaml.org/spec/1.1/#id857577.
    However, some YAML-generating tools generate a single document followed by
    this separator before ending the file.

    This method supports the case of a single document in a file that contains
    superfluous document separators, but still throws if multiple documents are
    actually found.

    Args:
      name: str, The file path to the file or "-" to read from stdin.

    Returns:
      The contents of the file parsed as a YAML data object.
    """
        from googlecloudsdk.core import yaml
        if name == '-':
            stdin = console_io.ReadStdin()
            yaml_data = yaml.load_all(stdin)
        else:
            yaml_data = yaml.load_all_path(name)
        yaml_data = [d for d in yaml_data if d is not None]
        if len(yaml_data) == 1:
            return yaml_data[0]
        if name == '-':
            return yaml.load(stdin)
        else:
            return yaml.load_path(name)

    def __call__(self, name):
        """Load YAML data from file path (name) or stdin.

    If name is "-", stdin is read until EOF. Otherwise, the named file is read.
    If self.validator is set, call it on the yaml data once it is loaded.

    Args:
      name: str, The file path to the file.

    Returns:
      The contents of the file parsed as a YAML data object.

    Raises:
      ArgumentTypeError: If the file cannot be read or is not a JSON/YAML like
        object.
      ValueError: If file content fails validation.
    """
        from googlecloudsdk.core import yaml
        try:
            yaml_data = self._LoadSingleYamlDocument(name)
            self._AssertJsonLike(yaml_data)
            if self.validator:
                if not self.validator(yaml_data):
                    raise ValueError('Invalid YAML/JSON content [{}]'.format(yaml_data))
            return yaml_data
        except (yaml.YAMLParseError, yaml.FileLoadError) as e:
            raise ArgumentTypeError(e)