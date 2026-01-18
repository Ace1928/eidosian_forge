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
class _ComponentBase(PyomoObject):
    """A base class for Component and ComponentData

    This class defines some fundamental methods and properties that are
    expected for all Component-like objects.  They are centralized here
    to avoid repeated code in the Component and ComponentData classes.
    """
    __slots__ = ()
    _PPRINT_INDENT = '    '

    def is_component_type(self):
        """Return True if this class is a Pyomo component"""
        return True

    def __deepcopy__(self, memo):
        if '__block_scope__' in memo:
            _scope = memo['__block_scope__']
            _new = None
            tmp = self.parent_block()
            _in_scope = tmp is None
            while id(tmp) not in _scope:
                _new = (_new, id(tmp))
                tmp = tmp.parent_block()
            _in_scope |= _scope[id(tmp)]
            while _new is not None:
                _new, _id = _new
                _scope[_id] = _in_scope
            if not _in_scope and id(self) not in _scope:
                memo[id(self)] = self
                return self
        component_list = []
        self._create_objects_for_deepcopy(memo, component_list)
        try:
            for i, comp in enumerate(component_list):
                saved_memo = len(memo)
                memo[id(comp)].__setstate__([fast_deepcopy(field, memo) for field in comp.__getstate__()])
            return memo[id(self)]
        except:
            pass
        for _ in range(len(memo) - saved_memo):
            memo.popitem()
        for comp in component_list[i:]:
            state = comp.__getstate__()
            _deepcopy_field = comp._deepcopy_field
            new_state = [_deepcopy_field(memo, slot, value) for slot, value in zip(comp.__auto_slots__.slots, state)]
            if comp.__auto_slots__.has_dict:
                new_state.append({slot: _deepcopy_field(memo, slot, value) for slot, value in state[-1].items()})
            memo[id(comp)].__setstate__(new_state)
        return memo[id(self)]

    def _create_objects_for_deepcopy(self, memo, component_list):
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append(self)
        return _ans

    def _deepcopy_field(self, memo, slot_name, value):
        saved_memo = len(memo)
        try:
            return fast_deepcopy(value, memo)
        except CloneError:
            raise
        except:
            for _ in range(len(memo) - saved_memo):
                memo.popitem()
            if '__block_scope__' not in memo:
                logger.warning("\n                    Uncopyable field encountered when deep\n                    copying outside the scope of Block.clone().\n                    There is a distinct possibility that the new\n                    copy is not complete.  To avoid this\n                    situation, either use Block.clone() or set\n                    'paranoid' mode by adding '__paranoid__' ==\n                    True to the memo before calling\n                    copy.deepcopy.")
            if self.model() is self:
                what = 'Model'
            else:
                what = 'Component'
            logger.error("Unable to clone Pyomo component attribute.\n%s '%s' contains an uncopyable field '%s' (%s).  Setting field to `None` on new object" % (what, self.name, slot_name, type(value)))
            if not self.parent_component()._constructed:
                raise CloneError('Uncopyable attribute (%s) encountered when cloning component %s on an abstract block.  The resulting instance is therefore missing data from the original abstract model and likely will not construct correctly.  Consider changing how you initialize this component or using a ConcreteModel.' % (slot_name, self.name))
        return None

    @deprecated('The cname() method has been renamed to getname().\n    The preferred method of obtaining a component name is to use the\n    .name property, which returns the fully qualified component name.\n    The .local_name property will return the component name only within\n    the context of the immediate parent container.', version='5.0')
    def cname(self, *args, **kwds):
        return self.getname(*args, **kwds)

    def pprint(self, ostream=None, verbose=False, prefix=''):
        """Print component information

        Note that this method is generally only reachable through
        ComponentData objects in an IndexedComponent container.
        Components, including unindexed Component derivatives and both
        scalar and indexed IndexedComponent derivatives will see
        :py:meth:`Component.pprint()`
        """
        comp = self.parent_component()
        _attr, _data, _header, _fcn = comp._pprint()
        if isinstance(type(_data), str):
            _name = comp.local_name
        else:
            _data = iter(((self.index(), self),))
            _name = '{Member of %s}' % (comp.local_name,)
        self._pprint_base_impl(ostream, verbose, prefix, _name, comp.doc, comp.is_constructed(), _attr, _data, _header, _fcn)

    @property
    def name(self):
        """Get the fully qualifed component name."""
        return self.getname(fully_qualified=True)

    @name.setter
    def name(self, val):
        raise ValueError('The .name attribute is now a property method that returns the fully qualified component name. Assignment is not allowed.')

    @property
    def local_name(self):
        """Get the component name only within the context of
        the immediate parent container."""
        return self.getname(fully_qualified=False)

    @property
    def active(self):
        """Return the active attribute"""
        return True

    @active.setter
    def active(self, value):
        """Set the active attribute to the given value"""
        raise AttributeError("Setting the 'active' flag on a component that does not support deactivation is not allowed.")

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