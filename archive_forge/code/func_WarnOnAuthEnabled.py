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
def WarnOnAuthEnabled(auth_enabled):
    """Adds prompt that describes lack of security provided by AUTH feature."""
    if auth_enabled:
        console_io.PromptContinue(message='AUTH prevents accidental access to the instance by ' + 'requiring an AUTH string (automatically generated for ' + 'you). AUTH credentials are not confidential when ' + 'transmitted or intended to protect against malicious ' + 'actors.', prompt_string='Do you want to proceed?', cancel_on_no=True)
    return auth_enabled