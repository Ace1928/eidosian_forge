from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceSplitResponse(_messages.Message):
    """The response to a SourceSplitRequest.

  Enums:
    OutcomeValueValuesEnum: Indicates whether splitting happened and produced
      a list of bundles. If this is USE_CURRENT_SOURCE_AS_IS, the current
      source should be processed "as is" without splitting. "bundles" is
      ignored in this case. If this is SPLITTING_HAPPENED, then "bundles"
      contains a list of bundles into which the source was split.

  Fields:
    bundles: If outcome is SPLITTING_HAPPENED, then this is a list of bundles
      into which the source was split. Otherwise this field is ignored. This
      list can be empty, which means the source represents an empty input.
    outcome: Indicates whether splitting happened and produced a list of
      bundles. If this is USE_CURRENT_SOURCE_AS_IS, the current source should
      be processed "as is" without splitting. "bundles" is ignored in this
      case. If this is SPLITTING_HAPPENED, then "bundles" contains a list of
      bundles into which the source was split.
    shards: DEPRECATED in favor of bundles.
  """

    class OutcomeValueValuesEnum(_messages.Enum):
        """Indicates whether splitting happened and produced a list of bundles.
    If this is USE_CURRENT_SOURCE_AS_IS, the current source should be
    processed "as is" without splitting. "bundles" is ignored in this case. If
    this is SPLITTING_HAPPENED, then "bundles" contains a list of bundles into
    which the source was split.

    Values:
      SOURCE_SPLIT_OUTCOME_UNKNOWN: The source split outcome is unknown, or
        unspecified.
      SOURCE_SPLIT_OUTCOME_USE_CURRENT: The current source should be processed
        "as is" without splitting.
      SOURCE_SPLIT_OUTCOME_SPLITTING_HAPPENED: Splitting produced a list of
        bundles.
    """
        SOURCE_SPLIT_OUTCOME_UNKNOWN = 0
        SOURCE_SPLIT_OUTCOME_USE_CURRENT = 1
        SOURCE_SPLIT_OUTCOME_SPLITTING_HAPPENED = 2
    bundles = _messages.MessageField('DerivedSource', 1, repeated=True)
    outcome = _messages.EnumField('OutcomeValueValuesEnum', 2)
    shards = _messages.MessageField('SourceSplitShard', 3, repeated=True)