import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def IsMacBundle(flavor, spec):
    """Returns if |spec| should be treated as a bundle.

  Bundles are directories with a certain subdirectory structure, instead of
  just a single file. Bundle rules do not produce a binary but also package
  resources into that directory."""
    is_mac_bundle = int(spec.get('mac_xctest_bundle', 0)) != 0 or int(spec.get('mac_xcuitest_bundle', 0)) != 0 or (int(spec.get('mac_bundle', 0)) != 0 and flavor == 'mac')
    if is_mac_bundle:
        assert spec['type'] != 'none', 'mac_bundle targets cannot have type none (target "%s")' % spec['target_name']
    return is_mac_bundle