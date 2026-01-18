import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _update_tag_positions(self, rule):
    """
        Update _tag_positions to reflect the changes to tags that are
        made by *rule*.
        """
    for pos in self._positions_by_rule[rule]:
        old_tag_positions = self._tag_positions[rule.original_tag]
        old_index = bisect.bisect_left(old_tag_positions, pos)
        del old_tag_positions[old_index]
        new_tag_positions = self._tag_positions[rule.replacement_tag]
        bisect.insort_left(new_tag_positions, pos)