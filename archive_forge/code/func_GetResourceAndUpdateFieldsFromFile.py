from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def GetResourceAndUpdateFieldsFromFile(file_path, resource_message_type):
    """Returns the resource message and update fields in file."""
    try:
        resource_to_parse = yaml.load_path(file_path)
    except yaml.YAMLParseError as e:
        raise exceptions.BadFileException('Policy config file [{0}] cannot be parsed. {1}'.format(file_path, six.text_type(e)))
    except yaml.FileLoadError as e:
        raise exceptions.BadFileException('Policy config file [{0}] cannot be opened or read. {1}'.format(file_path, six.text_type(e)))
    if not isinstance(resource_to_parse, dict):
        raise exceptions.BadFileException('Policy config file [{0}] is not a properly formatted YAML or JSON file.'.format(file_path))
    update_fields = list(resource_to_parse.keys())
    try:
        resource = encoding.PyValueToMessage(resource_message_type, resource_to_parse)
    except AttributeError as e:
        raise exceptions.BadFileException('Policy config file [{0}] is not a properly formatted YAML or JSON file. {1}'.format(file_path, six.text_type(e)))
    return (resource, update_fields)