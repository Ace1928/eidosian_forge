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