from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import redis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
import six
def GetMessagesForResource(resource_ref):
    api_version = resource_ref.GetCollectionInfo().api_version
    messages = redis.Messages(api_version)
    return messages