import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _connect_jump_to_finally_sections(self, node):
    """Connects a jump node to the finally sections protecting it."""
    cursor = set((node,))
    if node not in self.finally_sections:
        return cursor
    for guard_section_id in self.finally_sections[node]:
        guard_begin, guard_ends = self.finally_section_subgraphs[guard_section_id]
        self._connect_nodes(cursor, guard_begin)
        cursor = guard_ends
    del self.finally_sections[node]
    return cursor