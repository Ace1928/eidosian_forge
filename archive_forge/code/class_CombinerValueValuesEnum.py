from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CombinerValueValuesEnum(_messages.Enum):
    """How to combine the results of multiple conditions to determine if an
    incident should be opened. If condition_time_series_query_language is
    present, this must be COMBINE_UNSPECIFIED.

    Values:
      COMBINE_UNSPECIFIED: An unspecified combiner.
      AND: Combine conditions using the logical AND operator. An incident is
        created only if all the conditions are met simultaneously. This
        combiner is satisfied if all conditions are met, even if they are met
        on completely different resources.
      OR: Combine conditions using the logical OR operator. An incident is
        created if any of the listed conditions is met.
      AND_WITH_MATCHING_RESOURCE: Combine conditions using logical AND
        operator, but unlike the regular AND option, an incident is created
        only if all conditions are met simultaneously on at least one
        resource.
    """
    COMBINE_UNSPECIFIED = 0
    AND = 1
    OR = 2
    AND_WITH_MATCHING_RESOURCE = 3