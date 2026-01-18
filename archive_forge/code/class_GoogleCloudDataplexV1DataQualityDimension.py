from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityDimension(_messages.Message):
    """A dimension captures data quality intent about a defined subset of the
  rules specified.

  Fields:
    name: The dimension name a rule belongs to. Supported dimensions are
      "COMPLETENESS", "ACCURACY", "CONSISTENCY", "VALIDITY", "UNIQUENESS",
      "INTEGRITY"
  """
    name = _messages.StringField(1)