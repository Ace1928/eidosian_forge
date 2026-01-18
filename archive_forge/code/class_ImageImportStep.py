from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageImportStep(_messages.Message):
    """ImageImportStep holds information about the image import step progress.

  Fields:
    adaptingOs: Adapting OS step.
    creatingImage: Creating image step.
    endTime: Output only. The time the step has ended.
    initializing: Initializing step.
    loadingSourceFiles: Loading source files step.
    startTime: Output only. The time the step has started.
  """
    adaptingOs = _messages.MessageField('AdaptingOSStep', 1)
    creatingImage = _messages.MessageField('CreatingImageStep', 2)
    endTime = _messages.StringField(3)
    initializing = _messages.MessageField('InitializingImageImportStep', 4)
    loadingSourceFiles = _messages.MessageField('LoadingImageSourceFilesStep', 5)
    startTime = _messages.StringField(6)