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
def GetIamAccountFormatValidator():
    """Checks that provided iam account identifier is valid."""
    return arg_parsers.RegexpValidator('^(.+@.+\\..+|[0-9]+)$', 'Not a valid service account identifier. It should be either a numeric string representing the unique_id or an email of the form: my-iam-account@somedomain.com or my-iam-account@PROJECT_ID.iam.gserviceaccount.com')