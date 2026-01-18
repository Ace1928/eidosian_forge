import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def _process_opt_level(opt_level):
    if opt_level not in ('0', '1', '2', '3', 'max'):
        msg = f"Environment variable `NUMBA_OPT` is set to an unsupported value '{opt_level}', supported values are 0, 1, 2, 3, and 'max'"
        raise ValueError(msg)
    else:
        return _OptLevel(opt_level)