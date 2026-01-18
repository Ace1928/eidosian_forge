from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadContextSelector(_messages.Message):
    """Determines which workloads a policy is applicable for.

  Fields:
    metadataSelectors: Required. A map of metadata label values used to select
      workloads. If multiple MetadataSelectors are provided, all
      MetadataSelectors must match in order for the policy to be applied to
      this workload. Therefore these selectors must be combined in an AND
      fashion.
  """
    metadataSelectors = _messages.MessageField('WorkloadContextSelectorMetadataSelector', 1, repeated=True)