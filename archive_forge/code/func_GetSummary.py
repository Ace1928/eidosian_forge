from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def GetSummary(self):
    """Builds a summary.

    Returns:
      str, a markdown summary
    """
    all_locations = set(self.found_terms_map.values())
    if lookup.CAPSULE not in all_locations:
        all_locations.add(lookup.CAPSULE)

    def _Equivalent(location, other_location):
        """Returns True if both locations correspond to same summary section."""
        if location == other_location:
            return True
        if len(location) != len(other_location):
            return False
        if location[:-1] != other_location[:-1]:
            return False
        equivalent = [lookup.NAME, lookup.CHOICES, lookup.DESCRIPTION]
        if location[-1] in equivalent and other_location[-1] in equivalent:
            return True
        return False
    for full_location in sorted(sorted(all_locations), key=_SummaryPriority):
        location = full_location.split(DOT)
        terms = {t for t, l in six.iteritems(self.found_terms_map) if _Equivalent(l.split(DOT), location) and t}
        if location[0] == lookup.FLAGS:
            self._AddFlagToSummary(location, terms)
        elif location[0] == lookup.POSITIONALS:
            self._AddPositionalToSummary(location, terms)
        elif lookup.PATH in location or lookup.NAME in location:
            continue
        else:
            self._AddGenericSectionToSummary(location, terms)
    summary = '\n'.join(self._lines)
    return Highlight(summary, self.found_terms_map.keys())