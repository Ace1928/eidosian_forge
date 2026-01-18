from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessorVersionAlias(_messages.Message):
    """Contains the alias and the aliased resource name of processor version.

  Fields:
    alias: The alias in the form of `processor_version` resource name.
    processorVersion: The resource name of aliased processor version.
  """
    alias = _messages.StringField(1)
    processorVersion = _messages.StringField(2)