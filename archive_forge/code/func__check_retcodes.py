import os
import re
import shutil
import tempfile
from datetime import datetime
from time import time
from typing import List, Optional
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import get_logger
def _check_retcodes(self) -> bool:
    """Returns ``True`` when all chains have retcode 0."""
    for code in self._retcodes:
        if code != 0:
            return False
    return True