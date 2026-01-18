from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidDeviceCatalog(_messages.Message):
    """The currently supported Android devices.

  Fields:
    models: The set of supported Android device models.
    runtimeConfiguration: The set of supported runtime configurations.
    versions: The set of supported Android OS versions.
  """
    models = _messages.MessageField('AndroidModel', 1, repeated=True)
    runtimeConfiguration = _messages.MessageField('AndroidRuntimeConfiguration', 2)
    versions = _messages.MessageField('AndroidVersion', 3, repeated=True)