import sys, os
import textwrap
def _share_option_mappings(self, parser):
    self._short_opt = parser._short_opt
    self._long_opt = parser._long_opt
    self.defaults = parser.defaults