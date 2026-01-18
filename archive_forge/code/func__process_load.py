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
def _process_load(cmd, _model, _data, _default, options=None):
    from pyomo.core import Set
    _cmd_len = len(cmd)
    _options = {}
    _options['filename'] = cmd[1]
    i = 2
    while cmd[i] != ':':
        _options[cmd[i]] = cmd[i + 2]
        i += 3
    i += 1
    _Index = (None, [])
    if type(cmd[i]) is tuple:
        _Index = (None, cmd[i])
        i += 1
    elif i + 1 < _cmd_len and cmd[i + 1] == '=':
        _Index = (cmd[i], cmd[i + 2])
        i += 3
    _smap = OrderedDict()
    while i < _cmd_len:
        if i + 2 < _cmd_len and cmd[i + 1] == '=':
            _smap[cmd[i + 2]] = cmd[i]
            i += 3
        else:
            _smap[cmd[i]] = cmd[i]
            i += 1
    if len(cmd) < 2:
        raise IOError("The 'load' command must specify a filename")
    options = Bunch(**_options)
    for key in options:
        if not key in ['range', 'filename', 'format', 'using', 'driver', 'query', 'table', 'user', 'password', 'database']:
            raise ValueError("Unknown load option '%s'" % key)
    global Filename
    Filename = options.filename
    global Lineno
    Lineno = 0
    if options.using is None:
        tmp = options.filename.split('.')[-1]
        data = DataManagerFactory(tmp)
        if data is None or isinstance(data, UnknownDataManager):
            raise ApplicationError("Data manager '%s' is not available." % tmp)
    else:
        try:
            data = DataManagerFactory(options.using)
        except:
            data = None
        if data is None or isinstance(data, UnknownDataManager):
            raise ApplicationError("Data manager '%s' is not available." % options.using)
    set_name = None
    symb_map = _smap
    if len(symb_map) == 0:
        raise IOError('Must specify at least one set or parameter name that will be loaded')
    _index = None
    index_name = _Index[0]
    _select = None
    _set = None
    if options.format == 'set' or options.format == 'set_array':
        if len(_smap) != 1:
            raise IOError("A single set name must be specified when using format '%s'" % options.format)
        set_name = list(_smap.keys())[0]
        _set = set_name
    _param = None
    if options.format == 'transposed_array' or options.format == 'array' or options.format == 'param':
        if len(_smap) != 1:
            raise IOError("A single parameter name must be specified when using format '%s'" % options.format)
    if options.format in ('transposed_array', 'array', 'param', None):
        if _Index[0] is None:
            _index = None
        else:
            _index = _Index[0]
        _param = []
        _select = list(_Index[1])
        for key in _smap:
            _param.append(_smap[key])
            _select.append(key)
    if options.format in ('transposed_array', 'array'):
        _select = None
    if not _param is None and len(_param) == 1 and (not _model is None) and isinstance(getattr(_model, _param[0]), Set):
        _select = None
        _set = _param[0]
        _param = None
        _index = None
    data.initialize(model=options.model, filename=options.filename, index=_index, index_name=index_name, param_name=symb_map, set=_set, param=_param, format=options.format, range=options.range, query=options.query, using=options.using, table=options.table, select=_select, user=options.user, password=options.password, database=options.database)
    data.open()
    try:
        data.read()
    except Exception:
        data.close()
        raise
    data.close()
    data.process(_model, _data, _default)