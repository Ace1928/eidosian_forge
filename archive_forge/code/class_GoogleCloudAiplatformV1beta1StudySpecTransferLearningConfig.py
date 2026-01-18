from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecTransferLearningConfig(_messages.Message):
    """This contains flag for manually disabling transfer learning for a study.
  The names of prior studies being used for transfer learning (if any) are
  also listed here.

  Fields:
    disableTransferLearning: Flag to to manually prevent vizier from using
      transfer learning on a new study. Otherwise, vizier will automatically
      determine whether or not to use transfer learning.
    priorStudyNames: Output only. Names of previously completed studies
  """
    disableTransferLearning = _messages.BooleanField(1)
    priorStudyNames = _messages.StringField(2, repeated=True)