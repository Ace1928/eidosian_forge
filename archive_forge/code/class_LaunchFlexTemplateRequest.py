from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LaunchFlexTemplateRequest(_messages.Message):
    """A request to launch a Cloud Dataflow job from a FlexTemplate.

  Fields:
    launchParameter: Required. Parameter to launch a job form Flex Template.
    validateOnly: If true, the request is validated but not actually executed.
      Defaults to false.
  """
    launchParameter = _messages.MessageField('LaunchFlexTemplateParameter', 1)
    validateOnly = _messages.BooleanField(2)