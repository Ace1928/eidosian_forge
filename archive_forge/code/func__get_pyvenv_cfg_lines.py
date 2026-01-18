import logging
import os
import re
import site
import sys
from typing import List, Optional
def _get_pyvenv_cfg_lines() -> Optional[List[str]]:
    """Reads {sys.prefix}/pyvenv.cfg and returns its contents as list of lines

    Returns None, if it could not read/access the file.
    """
    pyvenv_cfg_file = os.path.join(sys.prefix, 'pyvenv.cfg')
    try:
        with open(pyvenv_cfg_file, encoding='utf-8') as f:
            return f.read().splitlines()
    except OSError:
        return None