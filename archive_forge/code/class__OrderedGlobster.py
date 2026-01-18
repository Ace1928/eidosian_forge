import re
from . import lazy_regex
from .trace import mutter, warning
class _OrderedGlobster(Globster):
    """A Globster that keeps pattern order."""

    def __init__(self, patterns):
        """Constructor.

        :param patterns: sequence of glob patterns
        """
        self._regex_patterns = []
        for pat in patterns:
            pat = normalize_pattern(pat)
            t = Globster.identify(pat)
            self._add_patterns([pat], Globster.pattern_info[t]['translator'], Globster.pattern_info[t]['prefix'])