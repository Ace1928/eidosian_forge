from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceSplitRequest(_messages.Message):
    """Represents the operation to split a high-level Source specification into
  bundles (parts for parallel processing). At a high level, splitting of a
  source into bundles happens as follows: SourceSplitRequest is applied to the
  source. If it returns SOURCE_SPLIT_OUTCOME_USE_CURRENT, no further splitting
  happens and the source is used "as is". Otherwise, splitting is applied
  recursively to each produced DerivedSource. As an optimization, for any
  Source, if its does_not_need_splitting is true, the framework assumes that
  splitting this source would return SOURCE_SPLIT_OUTCOME_USE_CURRENT, and
  doesn't initiate a SourceSplitRequest. This applies both to the initial
  source being split and to bundles produced from it.

  Fields:
    options: Hints for tuning the splitting process.
    source: Specification of the source to be split.
  """
    options = _messages.MessageField('SourceSplitOptions', 1)
    source = _messages.MessageField('Source', 2)