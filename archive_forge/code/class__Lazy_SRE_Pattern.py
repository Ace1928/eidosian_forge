from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.util import lazy_regex_patterns
class _Lazy_SRE_Pattern(object):
    """A class to lazily compile regex."""

    def __init__(self, pattern, flags=0):
        object.__setattr__(self, 'pattern', pattern)
        object.__setattr__(self, 'flags', flags)
        object.__setattr__(self, 'sre_pattern', None)

    def _compile(self):
        sre_pattern = real_compile(self.pattern, self.flags)
        object.__setattr__(self, 'sre_pattern', sre_pattern)

    def __getattr__(self, name):
        self._compile()
        return getattr(self.sre_pattern, name)

    def __setattr__(self, name, value):
        self._compile()
        setattr(self.sre_pattern, name, value)