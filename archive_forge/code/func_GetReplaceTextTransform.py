from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetReplaceTextTransform(value):
    replace_config = _GetMessageClass('GooglePrivacyDlpV2ReplaceValueConfig')
    value_holder = _GetMessageClass('GooglePrivacyDlpV2Value')
    return replace_config(newValue=value_holder(stringValue=value))