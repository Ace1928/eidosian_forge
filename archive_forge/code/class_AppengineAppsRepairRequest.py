from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsRepairRequest(_messages.Message):
    """A AppengineAppsRepairRequest object.

  Fields:
    name: Name of the application to repair. Example: apps/myapp
    repairApplicationRequest: A RepairApplicationRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    repairApplicationRequest = _messages.MessageField('RepairApplicationRequest', 2)