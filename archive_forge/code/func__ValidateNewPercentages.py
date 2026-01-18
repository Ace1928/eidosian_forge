from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _ValidateNewPercentages(self, new_percentages, unspecified_targets):
    """Validate the new traffic percentages the user specified."""
    specified_percent = sum(new_percentages.values())
    if specified_percent > 100:
        raise InvalidTrafficSpecificationError('Over 100% of traffic is specified.')
    for key in new_percentages:
        if new_percentages[key] < 0 or new_percentages[key] > 100:
            raise InvalidTrafficSpecificationError('New traffic for target %s is %s, not between 0 and 100' % (key, new_percentages[key]))
    if not unspecified_targets and specified_percent < 100:
        raise InvalidTrafficSpecificationError('Every target with traffic is updated but 100% of traffic has not been specified.')