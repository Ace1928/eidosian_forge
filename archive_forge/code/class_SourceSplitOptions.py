from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceSplitOptions(_messages.Message):
    """Hints for splitting a Source into bundles (parts for parallel
  processing) using SourceSplitRequest.

  Fields:
    desiredBundleSizeBytes: The source should be split into a set of bundles
      where the estimated size of each is approximately this many bytes.
    desiredShardSizeBytes: DEPRECATED in favor of desired_bundle_size_bytes.
  """
    desiredBundleSizeBytes = _messages.IntegerField(1)
    desiredShardSizeBytes = _messages.IntegerField(2)