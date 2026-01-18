import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _find_matching_unannotated_lines(output_lines, plain_child_lines, child_lines, start_child, end_child, right_lines, start_right, end_right, heads_provider, revision_id):
    """Find lines in plain_right_lines that match the existing lines.

    :param output_lines: Append final annotated lines to this list
    :param plain_child_lines: The unannotated new lines for the child text
    :param child_lines: Lines for the child text which have been annotated
        for the left parent

    :param start_child: Position in plain_child_lines and child_lines to start
        the match searching
    :param end_child: Last position in plain_child_lines and child_lines to
        search for a match
    :param right_lines: The annotated lines for the whole text for the right
        parent
    :param start_right: Position in right_lines to start the match
    :param end_right: Last position in right_lines to search for a match
    :param heads_provider: When parents disagree on the lineage of a line, we
        need to check if one side supersedes the other
    :param revision_id: The label to give if a line should be labeled 'tip'
    """
    output_extend = output_lines.extend
    output_append = output_lines.append
    plain_right_subset = [l for a, l in right_lines[start_right:end_right]]
    plain_child_subset = plain_child_lines[start_child:end_child]
    match_blocks = _get_matching_blocks(plain_right_subset, plain_child_subset)
    last_child_idx = 0
    for right_idx, child_idx, match_len in match_blocks:
        if child_idx > last_child_idx:
            output_extend(child_lines[start_child + last_child_idx:start_child + child_idx])
        for offset in range(match_len):
            left = child_lines[start_child + child_idx + offset]
            right = right_lines[start_right + right_idx + offset]
            if left[0] == right[0]:
                output_append(left)
            elif left[0] == revision_id:
                output_append(right)
            elif heads_provider is None:
                output_append((revision_id, left[1]))
            else:
                heads = heads_provider.heads((left[0], right[0]))
                if len(heads) == 1:
                    output_append((next(iter(heads)), left[1]))
                elif _break_annotation_tie is None:
                    output_append(_old_break_annotation_tie([left, right]))
                else:
                    output_append(_break_annotation_tie([left, right]))
        last_child_idx = child_idx + match_len