from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def _ConditionFileFormatException(filename):
    return gcloud_exceptions.InvalidArgumentException('condition-from-file', '{filename} must be a path to a YAML or JSON file containing the condition. `expression` and `title` are required keys. `description` is optional.'.format(filename=filename))