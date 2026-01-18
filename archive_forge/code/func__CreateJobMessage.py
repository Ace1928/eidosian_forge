from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.batch import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.batch import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
@classmethod
def _CreateJobMessage(cls, batch_msgs, config):
    """Parse into Job message using the config input.

    Args:
         batch_msgs: Batch defined proto message.
         config: The input content being either YAML or JSON or the HEREDOC
           input.

    Returns:
         The Parsed job message.
    """
    try:
        result = encoding.PyValueToMessage(batch_msgs.Job, yaml.load(config))
    except (ValueError, AttributeError, yaml.YAMLParseError):
        try:
            result = encoding.JsonToMessage(batch_msgs.Job, config)
        except (ValueError, DecodeError) as e:
            raise exceptions.Error('Unable to parse config file: {}'.format(e))
    return result