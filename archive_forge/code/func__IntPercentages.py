from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _IntPercentages(self, float_percentages):
    """Returns rounded integer percentages."""
    rounded_percentages = {k: int(float_percentages[k]) for k in float_percentages}
    loss = int(round(sum(float_percentages.values()))) - sum(rounded_percentages.values())
    correction_precedence = sorted(float_percentages.items(), key=NewRoundingCorrectionPrecedence)
    for key, _ in correction_precedence[:loss]:
        rounded_percentages[key] += 1
    return rounded_percentages