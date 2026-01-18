from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _resolve_annotation_tie(self, the_heads, line, tiebreaker):
    if tiebreaker is None:
        head = sorted(the_heads)[0]
    else:
        next_head = iter(the_heads)
        head = next(next_head)
        for possible_head in next_head:
            annotated_lines = ((head, line), (possible_head, line))
            head = tiebreaker(annotated_lines)[0]
    return head