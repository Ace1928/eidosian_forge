from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1BigQueryRoutineSpec(_messages.Message):
    """Fields specific for BigQuery routines.

  Fields:
    importedLibraries: Paths of the imported libraries.
  """
    importedLibraries = _messages.StringField(1, repeated=True)