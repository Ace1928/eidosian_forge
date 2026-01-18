from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExactMatchSpec(_messages.Message):
    """Spec for exact match metric - returns 1 if prediction and reference
  exactly matches, otherwise 0.
  """