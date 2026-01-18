from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ProfileConfigAbuse(_messages.Message):
    """Checks for abuse, which includes any requests sent to the API for
  purposes other than what it is intended for, such as high volumes of
  requests, data scraping, and abuse related to authorization.
  """