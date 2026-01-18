from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageTestEnabledRequest(_messages.Message):
    """A ServiceusageTestEnabledRequest object.

  Fields:
    name: Required. Resource name to check the value against hierarchically.
      Format: `projects/100`, `folders/101` or `organizations/102`.
    testEnabledRequest: A TestEnabledRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    testEnabledRequest = _messages.MessageField('TestEnabledRequest', 2)