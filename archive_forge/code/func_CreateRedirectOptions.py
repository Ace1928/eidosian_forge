from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateRedirectOptions(client, args):
    """Returns a SecurityPolicyRuleRedirectOptions message."""
    messages = client.messages
    redirect_options = messages.SecurityPolicyRuleRedirectOptions()
    is_updated = False
    if args.IsSpecified('redirect_type'):
        redirect_options.type = messages.SecurityPolicyRuleRedirectOptions.TypeValueValuesEnum(_ConvertRedirectType(args.redirect_type))
        is_updated = True
    if args.IsSpecified('redirect_target'):
        redirect_options.target = args.redirect_target
        if redirect_options.type is None:
            redirect_options.type = messages.SecurityPolicyRuleRedirectOptions.TypeValueValuesEnum.EXTERNAL_302
        is_updated = True
    return redirect_options if is_updated else None