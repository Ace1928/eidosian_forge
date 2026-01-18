from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def _DoesCommandRequireAppEngineApp():
    """Returns whether the command being executed needs App Engine app."""
    gcloud_env_key = constants.GCLOUD_COMMAND_ENV_KEY
    if gcloud_env_key in os.environ:
        return os.environ[gcloud_env_key] in constants.COMMANDS_THAT_NEED_APPENGINE
    return False