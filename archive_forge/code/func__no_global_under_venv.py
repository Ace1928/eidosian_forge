import logging
import os
import re
import site
import sys
from typing import List, Optional
def _no_global_under_venv() -> bool:
    """Check `{sys.prefix}/pyvenv.cfg` for system site-packages inclusion

    PEP 405 specifies that when system site-packages are not supposed to be
    visible from a virtual environment, `pyvenv.cfg` must contain the following
    line:

        include-system-site-packages = false

    Additionally, log a warning if accessing the file fails.
    """
    cfg_lines = _get_pyvenv_cfg_lines()
    if cfg_lines is None:
        logger.warning("Could not access 'pyvenv.cfg' despite a virtual environment being active. Assuming global site-packages is not accessible in this environment.")
        return True
    for line in cfg_lines:
        match = _INCLUDE_SYSTEM_SITE_PACKAGES_REGEX.match(line)
        if match is not None and match.group('value') == 'false':
            return True
    return False