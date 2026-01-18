from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _GetNormalizedTraffic(self):
    """Returns normalized targets, split into percent and tags targets.

    Moves all tags to 0% targets. Combines all targets with a non-zero percent
    that reference the same revision into a single target. Drops 0% targets
    without tags. Does not modify the underlying repeated message field.

    Returns:
      A tuple of (percent targets, tag targets), where percent targets is a
      dictionary mapping key to traffic target for all targets with percent
      greater than zero, and tag targets is a list of traffic targets with
      tags and percent equal to zero.
    """
    tag_targets = []
    percent_targets = {}
    for target in self._m:
        key = GetKey(target)
        if target.tag:
            tag_targets.append(NewTrafficTarget(self._messages, key, tag=target.tag))
        if target.percent:
            percent_targets.setdefault(key, NewTrafficTarget(self._messages, key, 0)).percent += target.percent
    return (percent_targets, tag_targets)