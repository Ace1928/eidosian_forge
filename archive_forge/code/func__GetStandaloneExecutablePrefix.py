import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetStandaloneExecutablePrefix(self):
    return self.spec.get('product_prefix', {'executable': '', 'static_library': 'lib', 'shared_library': 'lib', 'loadable_module': ''}[self.spec['type']])