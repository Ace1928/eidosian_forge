from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageFreshnessCheck(_messages.Message):
    """An image freshness check, which rejects images that were uploaded before
  the set number of days ago to the supported repositories.

  Fields:
    maxUploadAgeDays: Required. The max number of days that is allowed since
      the image was uploaded. Must be greater than zero.
  """
    maxUploadAgeDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)