from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateExpressionOptions(client, args):
    """Returns a SecurityPolicyRuleMatcherExprOptions message."""
    if not args.IsSpecified('recaptcha_action_site_keys') and (not args.IsSpecified('recaptcha_session_site_keys')):
        return None
    messages = client.messages
    recaptcha_options = messages.SecurityPolicyRuleMatcherExprOptionsRecaptchaOptions()
    if args.IsSpecified('recaptcha_action_site_keys'):
        recaptcha_options.actionTokenSiteKeys = args.recaptcha_action_site_keys
    if args.IsSpecified('recaptcha_session_site_keys'):
        recaptcha_options.sessionTokenSiteKeys = args.recaptcha_session_site_keys
    return messages.SecurityPolicyRuleMatcherExprOptions(recaptchaOptions=recaptcha_options)