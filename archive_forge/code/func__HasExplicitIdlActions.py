import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _HasExplicitIdlActions(self, spec):
    """Determine if an action should not run midl for .idl files."""
    return any([action.get('explicit_idl_action', 0) for action in spec.get('actions', [])])