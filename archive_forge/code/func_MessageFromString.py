from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def MessageFromString(msg_string, message_type, display_type, field_remappings=None, field_deletions=None):
    try:
        msg_as_yaml = yaml.load(msg_string)
        if field_remappings:
            msg_as_yaml = _RemapFields(msg_as_yaml, field_remappings)
        if field_deletions:
            msg_as_yaml = _DeleteFields(msg_as_yaml, field_deletions)
        msg = encoding.PyValueToMessage(message_type, msg_as_yaml)
        return msg
    except Exception as exc:
        raise YamlOrJsonLoadError('Could not parse YAML or JSON string for [{0}]: {1}'.format(display_type, exc))