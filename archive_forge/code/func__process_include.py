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
def _process_include(cmd, _model, _data, _default, options=None):
    if len(cmd) == 1:
        raise IOError("Cannot execute 'include' command without a filename")
    if len(cmd) > 2:
        raise IOError("The 'include' command only accepts a single filename")
    global Filename
    Filename = cmd[1]
    global Lineno
    Lineno = 0
    try:
        scenarios = parse_data_commands(filename=cmd[1])
    except IOError:
        raise
        err = sys.exc_info()[1]
        raise IOError("Error parsing file '%s': %s" % (Filename, str(err)))
    if scenarios is None:
        return False
    for scenario in scenarios:
        for cmd in scenarios[scenario]:
            if scenario not in _data:
                _data[scenario] = {}
            if cmd[0] in ('include', 'load'):
                _tmpdata = {}
                _process_data(cmd, _model, _tmpdata, _default, Filename, Lineno)
                if scenario is None:
                    for key in _tmpdata:
                        if key in _data:
                            _data[key].update(_tmpdata[key])
                        else:
                            _data[key] = _tmpdata[key]
                else:
                    for key in _tmpdata:
                        if key is None:
                            _data[scenario].update(_tmpdata[key])
                        else:
                            raise IOError('Cannot define a scenario within another scenario')
            else:
                _process_data(cmd, _model, _data[scenario], _default, Filename, Lineno)
    return True