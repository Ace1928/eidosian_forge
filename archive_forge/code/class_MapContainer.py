import copy
import enum
from io import StringIO
from math import inf
from pyomo.common.collections import Bunch
class MapContainer(dict):

    def __getnewargs_ex__(self):
        return ((0, 0), {})

    def __getnewargs__(self):
        return (0, 0)

    def __new__(cls, *args, **kwargs):
        _instance = super(MapContainer, cls).__new__(cls, *args, **kwargs)
        if len(args) > 1:
            super(MapContainer, _instance).__setattr__('_order', [])
        return _instance

    def __init__(self, ordered=False):
        dict.__init__(self)
        self._active = True
        self._required = False
        self._ordered = ordered
        self._order = []
        self._option = default_print_options

    def keys(self):
        return self._order

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except:
            pass
        try:
            self._active = True
            return self[self._convert(name)]
        except Exception:
            pass
        raise AttributeError('Unknown attribute `' + str(name) + "' for object with type " + str(type(self)))

    def __setattr__(self, name, val):
        if name == '__class__':
            self.__class__ = val
            return
        if name[0] == '_':
            self.__dict__[name] = val
            return
        self._active = True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError('Unknown attribute `' + str(name) + "' for object with type " + str(type(self)))
            self.declare(tmp)
        self._set_value(tmp, val)

    def __setitem__(self, name, val):
        self._active = True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError('Unknown attribute `' + str(name) + "' for object with type " + str(type(self)))
            self.declare(tmp)
        self._set_value(tmp, val)

    def _set_value(self, name, val):
        if isinstance(val, ListContainer) or isinstance(val, MapContainer):
            dict.__setitem__(self, name, val)
        elif isinstance(val, ScalarData):
            dict.__getitem__(self, name).value = val.value
        else:
            dict.__getitem__(self, name).value = val

    def __getitem__(self, name):
        tmp = self._convert(name)
        if tmp not in self:
            raise AttributeError('Unknown attribute `' + str(name) + "' for object with type " + str(type(self)))
        item = dict.__getitem__(self, tmp)
        if isinstance(item, ListContainer) or isinstance(item, MapContainer):
            return item
        return item.value

    def declare(self, name, **kwds):
        if name in self or type(name) is int:
            return
        tmp = self._convert(name)
        self._order.append(tmp)
        if 'value' in kwds and (isinstance(kwds['value'], MapContainer) or isinstance(kwds['value'], ListContainer)):
            if 'active' in kwds:
                kwds['value']._active = kwds['active']
            if 'required' in kwds and kwds['required'] is True:
                kwds['value']._required = True
            dict.__setitem__(self, tmp, kwds['value'])
        else:
            data = ScalarData(**kwds)
            if 'required' in kwds and kwds['required'] is True:
                data._required = True
            dict.__setitem__(self, tmp, data)

    def _repn_(self, option):
        if not option.schema and (not self._active) and (not self._required):
            return ignore
        if self._ordered:
            tmp = []
            for key in self._order:
                rep = dict.__getitem__(self, key)._repn_(option)
                if not rep == ignore:
                    tmp.append({key: rep})
        else:
            tmp = {}
            for key in self.keys():
                rep = dict.__getitem__(self, key)._repn_(option)
                if not rep == ignore:
                    tmp[key] = rep
        return tmp

    def _convert(self, name):
        if not isinstance(name, str):
            return name
        tmp = name.replace('_', ' ')
        return tmp[0].upper() + tmp[1:]

    def __repr__(self):
        return str(self._repn_(self._option))

    def __str__(self):
        ostream = StringIO()
        option = default_print_options
        self.pprint(ostream, self._option, repn=self._repn_(self._option))
        return ostream.getvalue()

    def pprint(self, ostream, option, from_list=False, prefix='', repn=None):
        if from_list:
            _prefix = ''
        else:
            _prefix = prefix
            ostream.write('\n')
        for key in self._order:
            if not key in repn:
                continue
            item = dict.__getitem__(self, key)
            ostream.write(_prefix + key + ': ')
            _prefix = prefix
            if isinstance(item, ListContainer):
                item.pprint(ostream, option, prefix=_prefix, repn=repn[key])
            else:
                item.pprint(ostream, option, prefix=_prefix + '  ', repn=repn[key])

    def load(self, repn):
        for key in repn:
            tmp = self._convert(key)
            if tmp not in self:
                self.declare(tmp)
            item = dict.__getitem__(self, tmp)
            item._active = True
            item.load(repn[key])

    def __getnewargs__(self):
        return (False, False)

    def __getstate__(self):
        return copy.copy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)