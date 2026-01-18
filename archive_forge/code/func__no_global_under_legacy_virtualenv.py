import logging
import os
import re
import site
import sys
from typing import List, Optional
def _no_global_under_legacy_virtualenv() -> bool:
    """Check if "no-global-site-packages.txt" exists beside site.py

    This mirrors logic in pypa/virtualenv for determining whether system
    site-packages are visible in the virtual environment.
    """
    site_mod_dir = os.path.dirname(os.path.abspath(site.__file__))
    no_global_site_packages_file = os.path.join(site_mod_dir, 'no-global-site-packages.txt')
    return os.path.exists(no_global_site_packages_file)