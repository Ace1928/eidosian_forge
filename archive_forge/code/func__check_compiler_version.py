from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
def _check_compiler_version(name):
    """Check if the installed version of the given compiler is greater than
    or equal to the required minimum version.
    """
    if name == 'catalyst':
        installed_catalyst_version = metadata.version('pennylane-catalyst')
        if Version(re.sub('\\.dev\\d+', '', installed_catalyst_version)) < PL_CATALYST_MIN_VERSION:
            raise CompileError(f'PennyLane-Catalyst {PL_CATALYST_MIN_VERSION} or greater is required, but installed {installed_catalyst_version}')