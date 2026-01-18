from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportSBOMRequest(_messages.Message):
    """The request to generate and export SBOM. Target must be specified for
  the request.

  Fields:
    cloudStorageLocation: Empty placeholder to denote that this is a Google
      Cloud Storage export request.
  """
    cloudStorageLocation = _messages.MessageField('CloudStorageLocation', 1)