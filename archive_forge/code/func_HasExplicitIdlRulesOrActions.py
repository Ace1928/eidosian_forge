import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def HasExplicitIdlRulesOrActions(self, spec):
    """Determine if there's an explicit rule or action for idl files. When
        there isn't we need to generate implicit rules to build MIDL .idl files."""
    return self._HasExplicitRuleForExtension(spec, 'idl') or self._HasExplicitIdlActions(spec)