from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def UpdateTags(self, to_update: Mapping[str, str], to_remove: Container[str], clear_others: bool):
    """Update traffic tags.

    Removes and/or clears existing traffic tags as requested. Always adds new
    tags to zero percent targets for the specified revision. Treats a tag
    update as a remove and add.

    Args:
      to_update: A dictionary mapping tag to revision name or 'LATEST' for the
        latest ready revision.
      to_remove: A list of tags to remove.
      clear_others: A boolean indicating whether to clear tags not specified in
        to_update.
    """
    new_targets = []
    if not self._m:
        self._m[:] = [NewTrafficTarget(self._messages, LATEST_REVISION_KEY, 100)]
    for target in self._m:
        if clear_others or target.tag in to_remove or target.tag in to_update:
            target.tag = None
        if target.percent or target.tag:
            new_targets.append(target)
    for tag, revision_key in sorted(to_update.items()):
        new_targets.append(NewTrafficTarget(self._messages, revision_key, tag=tag))
    self._m[:] = new_targets