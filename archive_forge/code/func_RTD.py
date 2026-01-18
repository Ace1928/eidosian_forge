import inspect
import io
import logging
import re
import sys
import textwrap
from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText
def RTD(_id):
    _id = str(_id).lower()
    assert _id[0] in 'wex'
    return f'{_RTD_URL}#{_id}'