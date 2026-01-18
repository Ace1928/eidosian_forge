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
def _apply_templates(cmd):
    template = []
    ilist = set()
    ans = []
    i = 0
    while i < len(cmd):
        if type(cmd[i]) is tuple and '*' in cmd[i]:
            j = i
            tmp = list(cmd[j])
            nindex = len(tmp)
            template = tmp
            ilist = set()
            for kk in range(nindex):
                if tmp[kk] == '*':
                    ilist.add(kk)
        elif len(ilist) == 0:
            ans.append(cmd[i])
        else:
            for kk in range(len(template)):
                if kk in ilist:
                    ans.append(cmd[i])
                    i += 1
                else:
                    ans.append(template[kk])
            ans.append(cmd[i])
        i += 1
    return ans