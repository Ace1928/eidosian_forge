from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsRedirectFunctionUpgradeTrafficRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsRedirectFunctionUpgradeTraffic
  Request object.

  Fields:
    name: Required. The name of the function for which traffic target should
      be changed to 2nd Gen from 1st Gen.
    redirectFunctionUpgradeTrafficRequest: A
      RedirectFunctionUpgradeTrafficRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    redirectFunctionUpgradeTrafficRequest = _messages.MessageField('RedirectFunctionUpgradeTrafficRequest', 2)