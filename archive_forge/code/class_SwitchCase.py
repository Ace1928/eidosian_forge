from __future__ import absolute_import
import types
from . import Errors
class SwitchCase(RE):
    """
    SwitchCase(re, nocase) is an RE which matches the same strings as RE,
    but treating upper and lower case letters according to |nocase|. If
    |nocase| is true, case is ignored, otherwise it is not.
    """
    re = None
    nocase = None

    def __init__(self, re, nocase):
        self.re = re
        self.nocase = nocase
        self.nullable = re.nullable
        self.match_nl = re.match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        self.re.build_machine(m, initial_state, final_state, match_bol, self.nocase)

    def calc_str(self):
        if self.nocase:
            name = 'NoCase'
        else:
            name = 'Case'
        return '%s(%s)' % (name, self.re)