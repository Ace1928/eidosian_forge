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
def _process_table(cmd, _model, _data, _default, options=None):
    _options = {}
    _set = OrderedDict()
    _param = OrderedDict()
    _labels = []
    _cmd = cmd[1]
    _cmd_len = len(_cmd)
    name = None
    i = 0
    while i < _cmd_len:
        try:
            if _cmd[i] == ':':
                i += 1
                while i < _cmd_len:
                    _labels.append(_cmd[i])
                    i += 1
                continue
            name = _cmd[i]
            if i + 1 == _cmd_len:
                _param[name] = []
                _labels = ['Z']
                i += 1
                continue
            if _cmd[i + 1] == '=':
                if type(_cmd[i + 2]) is list:
                    _set[name] = _cmd[i + 2]
                else:
                    _options[name] = _cmd[i + 2]
                i += 3
                continue
            if not type(_cmd[i + 1]) is tuple:
                raise IOError
            if i + 2 < _cmd_len and _cmd[i + 2] == '=':
                _param[name] = (_cmd[i + 1], _cmd[i + 3][0])
                i += 4
            else:
                _param[name] = _cmd[i + 1]
                i += 2
        except:
            raise IOError('Error parsing table options: %s' % name)
    options = Bunch(**_options)
    for key in options:
        if not key in ['columns']:
            raise ValueError("Unknown table option '%s'" % key)
    ncolumns = options.columns
    if ncolumns is None:
        ncolumns = len(_labels)
        if ncolumns == 0:
            if not (len(_set) == 1 and len(_set[_set.keys()[0]]) == 0):
                raise IOError("Must specify either the 'columns' option or column headers")
            else:
                ncolumns = 1
    else:
        ncolumns = int(ncolumns)
    data = cmd[2]
    Ldata = len(cmd[2])
    cmap = {}
    if len(_labels) == 0:
        for i in range(ncolumns):
            cmap[i + 1] = i
        for label in _param:
            ndx = cmap[_param[label][1]]
            if ndx < 0 or ndx >= ncolumns:
                raise IOError('Bad column value %s for data %s' % (str(ndx), label))
            cmap[label] = ndx
            _param[label] = _param[label][0]
    else:
        i = 0
        for label in _labels:
            cmap[label] = i
            i += 1
    for sname in _set:
        cols = _set[sname]
        tmp = []
        for col in cols:
            if not col in cmap:
                raise IOError("Unexpected table column '%s' for index set '%s'" % (col, sname))
            tmp.append(cmap[col])
        if not sname in cmap:
            cmap[sname] = tmp
        cols = list(flatten_tuple(tmp))
        _cmd = ['set', sname, ':=']
        i = 0
        while i < Ldata:
            row = []
            for col in cols:
                row.append(data[i + col])
            if len(row) > 1:
                _cmd.append(tuple(row))
            else:
                _cmd.append(row[0])
            i += ncolumns
        _process_set(_cmd, _model, _data)
    _i = 0
    if ncolumns == 0:
        raise IOError
    for vname in _param:
        _i += 1
        cols = _param[vname]
        tmp = []
        for col in cols:
            if not col in cmap:
                raise IOError("Unexpected table column '%s' for table value '%s'" % (col, vname))
            tmp.append(cmap[col])
        cols = list(flatten_tuple(tmp))
        if vname in cmap:
            cols.append(cmap[vname])
        else:
            cols.append(ncolumns - 1 - (len(_param) - _i))
        _cmd = ['param', vname, ':=']
        i = 0
        while i < Ldata:
            for col in cols:
                _cmd.append(data[i + col])
            i += ncolumns
        _process_param(_cmd, _model, _data, None, ncolumns=len(cols))