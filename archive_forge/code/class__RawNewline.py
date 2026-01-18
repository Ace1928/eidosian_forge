from __future__ import absolute_import
import types
from . import Errors
class _RawNewline(RE):
    """
    RawNewline is a low-level RE which matches a newline character.
    For internal use only.
    """
    nullable = 0
    match_nl = 1

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if match_bol:
            initial_state = self.build_opt(m, initial_state, BOL)
        s = self.build_opt(m, initial_state, EOL)
        s.add_transition((nl_code, nl_code + 1), final_state)