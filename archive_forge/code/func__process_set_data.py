import sys
import re
import copy
import logging
from pyomo.common.log import is_debug_set
from pyomo.common.collections import Bunch, OrderedDict
from pyomo.common.errors import ApplicationError
from pyomo.dataportal.parse_datacmds import parse_data_commands, _re_number
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.util import flatten_tuple
def _process_set_data(cmd, sname, _model):
    """
    Called by _process_set() to process set data.
    """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug('DEBUG: _process_set_data(start) %s', cmd)
    if len(cmd) == 0:
        return []
    ans = []
    i = 0
    template = None
    ndx = []
    template = []
    while i < len(cmd):
        if type(cmd[i]) is not tuple:
            if len(ndx) == 0:
                ans.append(cmd[i])
            else:
                tmpval = template
                for kk in range(len(ndx)):
                    if i == len(cmd):
                        raise IOError('Expected another set value to flush out a tuple pattern!')
                    tmpval[ndx[kk]] = cmd[i]
                    i += 1
                ans.append(tuple(tmpval))
                continue
        elif '*' not in cmd[i]:
            ans.append(cmd[i])
        else:
            template = list(cmd[i])
            ndx = []
            for kk in range(len(template)):
                if template[kk] == '*':
                    ndx.append(kk)
        i += 1
    if generate_debug_messages:
        logger.debug('DEBUG: _process_set_data(end) %s', ans)
    return ans