from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteAdminQuotaPolicyMetadata(_messages.Message):
    """Metadata message that provides information such as progress, partial
  failures, and similar information on each GetOperation call of LRO returned
  by DeleteAdminQuotaPolicy.
  """