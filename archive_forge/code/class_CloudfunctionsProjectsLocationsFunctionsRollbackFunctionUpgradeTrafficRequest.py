from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsRollbackFunctionUpgradeTrafficRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsRollbackFunctionUpgradeTraffic
  Request object.

  Fields:
    name: Required. The name of the function for which traffic target should
      be changed back to 1st Gen from 2nd Gen.
    rollbackFunctionUpgradeTrafficRequest: A
      RollbackFunctionUpgradeTrafficRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackFunctionUpgradeTrafficRequest = _messages.MessageField('RollbackFunctionUpgradeTrafficRequest', 2)