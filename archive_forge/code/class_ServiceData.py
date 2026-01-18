from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceData(_messages.Message):
    """This message defines service-specific data that certain service teams
  must provide as part of the Data Residency Augmented View for a resource.
  Next ID: 2

  Fields:
    pd: Auxiliary data for the persistent disk pipeline provided to provide
      the LSV Colossus Roots and GCS Buckets.
  """
    pd = _messages.MessageField('PersistentDiskData', 1)