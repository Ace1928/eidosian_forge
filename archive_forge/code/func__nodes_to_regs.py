from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def _nodes_to_regs(self):
    """
        Return a list of (varname, reg) tuples.
        """

    def get_tuples():
        for r, re_match in self._re_matches:
            for group_name, group_index in r.groupindex.items():
                if group_name != _INVALID_TRAILING_INPUT:
                    reg = re_match.regs[group_index]
                    node = self._group_names_to_nodes[group_name]
                    yield (node, reg)
    return list(get_tuples())