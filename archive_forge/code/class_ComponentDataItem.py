import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
class ComponentDataItem(object):
    """
    This is a container for a Pyomo component to be displayed in a model tree
    view.

    Args:
        parent: parent data item
        o: pyomo component object
        ui_data: a container for data, as of now mainly just pyomo model
    """

    def __init__(self, parent, o, ui_data):
        self.ui_data = ui_data
        self.data = o
        self.parent = parent
        self.children = []
        self.ids = {}
        self.get_callback = {'value': self._get_value_callback, 'lb': self._get_lb_callback, 'ub': self._get_ub_callback, 'expr': self._get_expr_callback, 'residual': self._get_residual_callback, 'units': self._get_units_callback, 'domain': self._get_domain_callback}
        self.set_callback = {'value': self._set_value_callback, 'lb': self._set_lb_callback, 'ub': self._set_ub_callback, 'active': self._set_active_callback, 'fixed': self._set_fixed_callback}

    @property
    def _cache_value(self):
        return self.ui_data.value_cache.get(self.data, None)

    @property
    def _cache_units(self):
        return self.ui_data.value_cache_units.get(self.data, None)

    def add_child(self, o):
        """Add a child data item"""
        item = ComponentDataItem(self, o, ui_data=self.ui_data)
        self.children.append(item)
        self.ids[id(o)] = item
        return item

    def get(self, a):
        """Get an attribute"""
        if a in self.get_callback:
            return self.get_callback[a]()
        else:
            try:
                return getattr(self.data, a)
            except:
                return None

    def set(self, a, val):
        """set an attribute"""
        if a in self.set_callback:
            return self.set_callback[a](val)
        else:
            try:
                return setattr(self.data, a, val)
            except:
                _log.exception("Can't set value of {}".format(a))
                return None

    def _get_expr_callback(self):
        if hasattr(self.data, 'expr'):
            return str(self.data.expr)
        else:
            return None

    def _get_value_callback(self):
        if isinstance(self.data, _ParamData):
            v = value_no_exception(self.data, div0='divide_by_0')
            if isinstance(v, float):
                v = float(v)
            elif isinstance(v, int):
                v = int(v)
            return v
        elif isinstance(self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)):
            v = value_no_exception(self.data)
            if isinstance(v, float):
                v = float(v)
            elif isinstance(v, int):
                v = int(v)
            return v
        elif isinstance(self.data, (float, int)):
            return self.data
        else:
            return self._cache_value

    def _get_lb_callback(self):
        if isinstance(self.data, Var._ComponentDataClass):
            return self.data.lb
        elif hasattr(self.data, 'lower'):
            return value_no_exception(self.data.lower, div0='Divide_by_0')
        else:
            return None

    def _get_ub_callback(self):
        if isinstance(self.data, Var._ComponentDataClass):
            return self.data.ub
        elif hasattr(self.data, 'upper'):
            return value_no_exception(self.data.upper, div0='Divide_by_0')
        else:
            return None

    def _get_residual_callback(self):
        if isinstance(self.data, Constraint._ComponentDataClass):
            return get_residual(self.ui_data, self.data)
        else:
            return None

    def _get_units_callback(self):
        if isinstance(self.data, (Var, Var._ComponentDataClass)):
            return str(units.get_units(self.data))
        if isinstance(self.data, (Param, _ParamData)):
            return str(units.get_units(self.data))
        return self._cache_units

    def _get_domain_callback(self):
        if isinstance(self.data, Var._ComponentDataClass):
            return str(self.data.domain)
        if isinstance(self.data, (BooleanVar, BooleanVar._ComponentDataClass)):
            return 'BooleanVar'
        return None

    def _set_value_callback(self, val):
        if isinstance(self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)):
            try:
                self.data.value = val
            except:
                return
        elif isinstance(self.data, (Var, BooleanVar)):
            try:
                for o in self.data.values():
                    o.value = val
            except:
                return
        elif isinstance(self.data, _ParamData):
            if not self.data.parent_component().mutable:
                return
            try:
                self.data.value = val
            except:
                return
        elif isinstance(self.data, Param):
            if not self.data.parent_component().mutable:
                return
            try:
                for o in self.data.values():
                    o.value = val
            except:
                return

    def _set_lb_callback(self, val):
        if isinstance(self.data, Var._ComponentDataClass):
            try:
                self.data.setlb(val)
            except:
                return
        elif isinstance(self.data, Var):
            try:
                for o in self.data.values():
                    o.setlb(val)
            except:
                return

    def _set_ub_callback(self, val):
        if isinstance(self.data, Var._ComponentDataClass):
            try:
                self.data.setub(val)
            except:
                return
        elif isinstance(self.data, Var):
            try:
                for o in self.data.values():
                    o.setub(val)
            except:
                return

    def _set_active_callback(self, val):
        if not val or val in ['False', 'false', '0', 'f', 'F']:
            val = False
        else:
            val = True
        try:
            if val:
                self.data.activate()
            else:
                self.data.deactivate()
        except:
            return

    def _set_fixed_callback(self, val):
        if not val or val in ['False', 'false', '0', 'f', 'F']:
            val = False
        else:
            val = True
        try:
            if val:
                self.data.fix()
            else:
                self.data.unfix()
        except:
            return