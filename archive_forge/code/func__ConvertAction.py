from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def _ConvertAction(self, action):
    return {'deny-403': 'deny(403)', 'deny-404': 'deny(404)', 'deny-502': 'deny(502)', 'redirect-to-recaptcha': 'redirect_to_recaptcha', 'rate-based-ban': 'rate_based_ban'}.get(action, action)