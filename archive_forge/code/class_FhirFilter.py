from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FhirFilter(_messages.Message):
    """Filter configuration.

  Fields:
    resources: List of resources to include in the output. If this list is
      empty or not specified, all resources are included in the output.
  """
    resources = _messages.MessageField('Resources', 1)