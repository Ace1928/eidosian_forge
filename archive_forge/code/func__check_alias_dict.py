import sys, string, re
import getopt
from distutils.errors import *
def _check_alias_dict(self, aliases, what):
    assert isinstance(aliases, dict)
    for alias, opt in aliases.items():
        if alias not in self.option_index:
            raise DistutilsGetoptError("invalid %s '%s': option '%s' not defined" % (what, alias, alias))
        if opt not in self.option_index:
            raise DistutilsGetoptError("invalid %s '%s': aliased option '%s' not defined" % (what, alias, opt))