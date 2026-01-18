from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import env_vars as env_vars_api_util
from googlecloudsdk.api_lib.functions.v1 import exceptions as function_exceptions
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as v2_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions.v1.deploy import enum_util
from googlecloudsdk.command_lib.functions.v1.deploy import labels_util
from googlecloudsdk.command_lib.functions.v1.deploy import source_util
from googlecloudsdk.command_lib.functions.v1.deploy import trigger_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import urllib
def TryToLogStackdriverURL(op):
    """Logs stackdriver URL.

    This is for executing in the polling loop, and will stop trying as soon as
    it succeeds at making a change.

    Args:
      op: the operation
    """
    if log_stackdriver_url[0] and op.metadata:
        metadata = encoding.PyValueToMessage(messages.OperationMetadataV1, encoding.MessageToPyValue(op.metadata))
        if metadata.buildName and _BUILD_NAME_REGEX.match(metadata.buildName):
            log.status.Print('\nFor Cloud Build Logs, visit: %s' % _CreateCloudBuildLogURL(metadata.buildName))
            log_stackdriver_url[0] = False
        elif metadata.buildId:
            sd_info_template = '\nFor Cloud Build Stackdriver Logs, visit: %s'
            log.status.Print(sd_info_template % _CreateStackdriverURLforBuildLogs(metadata.buildId, _GetProject()))
            log_stackdriver_url[0] = False