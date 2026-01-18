from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HelloWorldFeatureSpec(_messages.Message):
    """**Hello World**: The Hub-wide input for the HelloWorld feature.

  Fields:
    customConfig: Custom config for the HelloWorld controller codelab. This
      should be a textpb string.
    featureSample: Message to hold fields to use in feature e2e create/mutate
      testing.
  """
    customConfig = _messages.StringField(1)
    featureSample = _messages.MessageField('HelloWorldFeatureSample', 2)