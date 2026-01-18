import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
def _pprint_base_impl(self, ostream, verbose, prefix, _name, _doc, _constructed, _attr, _data, _header, _fcn):
    if ostream is None:
        ostream = sys.stdout
    if prefix:
        ostream = StreamIndenter(ostream, prefix)
    if not _attr and self.parent_block() is None:
        _name = ''
    if _attr or _name or _doc:
        ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
        ostream.newline = False
    if self.is_reference():
        _attr = list(_attr) if _attr else []
        _attr.append(('ReferenceTo', self.referent))
    if _name:
        ostream.write(_name + ' : ')
    if _doc:
        ostream.write(_doc + '\n')
    if _attr:
        ostream.write(', '.join(('%s=%s' % (k, v) for k, v in _attr)))
    if _attr or _name or _doc:
        ostream.write('\n')
    if not _constructed:
        if self.parent_block() is not None:
            ostream.write('Not constructed\n')
            return
    if type(_fcn) is tuple:
        _fcn, _fcn2 = _fcn
    else:
        _fcn2 = None
    if _header is not None:
        if _fcn2 is not None:
            _data_dict = dict(_data)
            _data = _data_dict.items()
        tabular_writer(ostream, '', _data, _header, _fcn)
        if _fcn2 is not None:
            for _key in sorted_robust(_data_dict):
                _fcn2(ostream, _key, _data_dict[_key])
    elif _fcn is not None:
        _data_dict = dict(_data)
        for _key in sorted_robust(_data_dict):
            _fcn(ostream, _key, _data_dict[_key])
    elif _data is not None:
        ostream.write(_data)