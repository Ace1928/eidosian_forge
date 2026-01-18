from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetHintForServiceAccountResource(action='act on'):
    """Returns a hint message for commands treating service account as a resource.

  Args:
    action: the action to take on the service account resource (with necessary
      prepositions), such as 'add iam policy bindings to'.
  """
    return 'When managing IAM roles, you can treat a service account either as a resource or as an identity. This command is to {action} a service account resource. There are other gcloud commands to manage IAM policies for other types of resources. For example, to manage IAM policies on a project, use the `$ gcloud projects` commands.'.format(action=action)