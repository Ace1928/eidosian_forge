from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsRuntimeVersionsGetRequest(_messages.Message):
    """A TpuProjectsLocationsRuntimeVersionsGetRequest object.

  Fields:
    name: Required. The resource name.
  """
    name = _messages.StringField(1, required=True)