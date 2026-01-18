import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
class BasePlotlyType(object):
    """
    BasePlotlyType is the base class for all objects in the trace, layout,
    and frame object hierarchies
    """
    _mapped_properties = {}
    _parent_path_str = ''
    _path_str = ''
    _valid_props = set()

    def __init__(self, plotly_name, **kwargs):
        """
        Construct a new BasePlotlyType

        Parameters
        ----------
        plotly_name : str
            The lowercase name of the plotly object
        kwargs : dict
            Invalid props/values to raise on
        """
        self._skip_invalid = False
        self._validate = True
        self._process_kwargs(**kwargs)
        self._plotly_name = plotly_name
        self._compound_props = {}
        self._compound_array_props = {}
        self._orphan_props = {}
        self._parent = None
        self._change_callbacks = {}
        self.__validators = None

    def _get_validator(self, prop):
        from .validator_cache import ValidatorCache
        return ValidatorCache.get_validator(self._path_str, prop)

    @property
    def _validators(self):
        """
        Validators used to be stored in a private _validators property. This was
        eliminated when we switched to building validators on demand using the
        _get_validator method.

        This property returns a simple object that

        Returns
        -------
        dict-like interface for accessing the object's validators
        """
        obj = self
        if self.__validators is None:

            class ValidatorCompat(object):

                def __getitem__(self, item):
                    return obj._get_validator(item)

                def __contains__(self, item):
                    return obj.__contains__(item)

                def __iter__(self):
                    return iter(obj)

                def items(self):
                    return [(k, self[k]) for k in self]
            self.__validators = ValidatorCompat()
        return self.__validators

    def _process_kwargs(self, **kwargs):
        """
        Process any extra kwargs that are not predefined as constructor params
        """
        for k, v in kwargs.items():
            err = _check_path_in_prop_tree(self, k, error_cast=ValueError)
            if err is None:
                self[k] = v
            elif not self._validate:
                self[k] = v
            elif not self._skip_invalid:
                raise err

    @property
    def plotly_name(self):
        """
        The plotly name of the object

        Returns
        -------
        str
        """
        return self._plotly_name

    @property
    def _prop_descriptions(self):
        """
        Formatted string containing all of this obejcts child properties
        and their descriptions

        Returns
        -------
        str
        """
        raise NotImplementedError

    @property
    def _props(self):
        """
        Dictionary used to store this object properties.  When the object
        has a parent, this dict is retreived from the parent. When the
        object does not have a parent, this dict is the object's
        `_orphan_props` property

        Note: Property will return None if the object has a parent and the
        object's properties have not been initialized using the
        `_init_props` method.

        Returns
        -------
        dict|None
        """
        if self.parent is None:
            return self._orphan_props
        else:
            return self.parent._get_child_props(self)

    def _get_child_props(self, child):
        """
        Return properties dict for child

        Parameters
        ----------
        child : BasePlotlyType

        Returns
        -------
        dict
        """
        if self._props is None:
            return None
        elif child.plotly_name in self:
            from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator
            validator = self._get_validator(child.plotly_name)
            if isinstance(validator, CompoundValidator):
                return self._props.get(child.plotly_name, None)
            elif isinstance(validator, CompoundArrayValidator):
                children = self[child.plotly_name]
                child_ind = BaseFigure._index_is(children, child)
                assert child_ind is not None
                children_props = self._props.get(child.plotly_name, None)
                return children_props[child_ind] if children_props is not None and len(children_props) > child_ind else None
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    def _init_props(self):
        """
        Ensure that this object's properties dict has been initialized. When
        the object has a parent, this ensures that the parent has an
        initialized properties dict with this object's plotly_name as a key.

        Returns
        -------
        None
        """
        if self._props is not None:
            pass
        else:
            self._parent._init_child_props(self)

    def _init_child_props(self, child):
        """
        Ensure that a properties dict has been initialized for a child object

        Parameters
        ----------
        child : BasePlotlyType

        Returns
        -------
        None
        """
        self._init_props()
        if child.plotly_name in self._compound_props:
            if child.plotly_name not in self._props:
                self._props[child.plotly_name] = {}
        elif child.plotly_name in self._compound_array_props:
            children = self._compound_array_props[child.plotly_name]
            child_ind = BaseFigure._index_is(children, child)
            assert child_ind is not None
            if child.plotly_name not in self._props:
                self._props[child.plotly_name] = []
            children_list = self._props[child.plotly_name]
            while len(children_list) <= child_ind:
                children_list.append({})
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    def _get_child_prop_defaults(self, child):
        """
        Return default properties dict for child

        Parameters
        ----------
        child : BasePlotlyType

        Returns
        -------
        dict
        """
        if self._prop_defaults is None:
            return None
        elif child.plotly_name in self._compound_props:
            return self._prop_defaults.get(child.plotly_name, None)
        elif child.plotly_name in self._compound_array_props:
            children = self._compound_array_props[child.plotly_name]
            child_ind = BaseFigure._index_is(children, child)
            assert child_ind is not None
            children_props = self._prop_defaults.get(child.plotly_name, None)
            return children_props[child_ind] if children_props is not None and len(children_props) > child_ind else None
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    @property
    def _prop_defaults(self):
        """
        Return default properties dict

        Returns
        -------
        dict
        """
        if self.parent is None:
            return None
        else:
            return self.parent._get_child_prop_defaults(self)

    def _get_prop_validator(self, prop):
        """
        Return the validator associated with the specified property

        Parameters
        ----------
        prop: str
            A property that exists in this object

        Returns
        -------
        BaseValidator
        """
        if prop in self._mapped_properties:
            prop_path = self._mapped_properties[prop]
            plotly_obj = self[prop_path[:-1]]
            prop = prop_path[-1]
        else:
            prop_path = BaseFigure._str_to_dict_path(prop)
            plotly_obj = self[prop_path[:-1]]
            prop = prop_path[-1]
        return plotly_obj._get_validator(prop)

    @property
    def parent(self):
        """
        Return the object's parent, or None if the object has no parent
        Returns
        -------
        BasePlotlyType|BaseFigure
        """
        return self._parent

    @property
    def figure(self):
        """
        Reference to the top-level Figure or FigureWidget that this object
        belongs to. None if the object does not belong to a Figure

        Returns
        -------
        Union[BaseFigure, None]
        """
        top_parent = self
        while top_parent is not None:
            if isinstance(top_parent, BaseFigure):
                break
            else:
                top_parent = top_parent.parent
        return top_parent

    def __reduce__(self):
        """
        Custom implementation of reduce is used to support deep copying
        and pickling
        """
        props = self.to_plotly_json()
        return (self.__class__, (props,))

    def __getitem__(self, prop):
        """
        Get item or nested item from object

        Parameters
        ----------
        prop : str|tuple

            If prop is the name of a property of this object, then the
            property is returned.

            If prop is a nested property path string (e.g. 'foo[1].bar'),
            then a nested property is returned (e.g. obj['foo'][1]['bar'])

            If prop is a path tuple (e.g. ('foo', 1, 'bar')), then a nested
            property is returned (e.g. obj['foo'][1]['bar']).

        Returns
        -------
        Any
        """
        from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator, BaseDataValidator
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if prop and prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
            orig_prop = _remake_path_from_tuple(prop)
        if len(prop) == 1:
            prop = prop[0]
            if prop not in self._valid_props:
                self._raise_on_invalid_property_error(_error_to_raise=PlotlyKeyError)(prop)
            validator = self._get_validator(prop)
            if isinstance(validator, CompoundValidator):
                if self._compound_props.get(prop, None) is None:
                    self._compound_props[prop] = validator.data_class(_parent=self, plotly_name=prop)
                    self._compound_props[prop]._plotly_name = prop
                return validator.present(self._compound_props[prop])
            elif isinstance(validator, (CompoundArrayValidator, BaseDataValidator)):
                if self._compound_array_props.get(prop, None) is None:
                    if self._props is not None:
                        self._compound_array_props[prop] = [validator.data_class(_parent=self) for _ in self._props.get(prop, [])]
                    else:
                        self._compound_array_props[prop] = []
                return validator.present(self._compound_array_props[prop])
            elif self._props is not None and prop in self._props:
                return validator.present(self._props[prop])
            elif self._prop_defaults is not None:
                return validator.present(self._prop_defaults.get(prop, None))
            else:
                return None
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=PlotlyKeyError)
            if err is not None:
                raise err
            res = self
            for p in prop:
                res = res[p]
            return res

    def __contains__(self, prop):
        """
        Determine whether object contains a property or nested property

        Parameters
        ----------
        prop : str|tuple
            If prop is a simple string (e.g. 'foo'), then return true of the
            object contains an element named 'foo'

            If prop is a property path string (e.g. 'foo[0].bar'),
            then return true if the obejct contains the nested elements for
            each entry in the path string (e.g. 'bar' in obj['foo'][0])

            If prop is a property path tuple (e.g. ('foo', 0, 'bar')),
            then return true if the object contains the nested elements for
            each entry in the path string (e.g. 'bar' in obj['foo'][0])

        Returns
        -------
        bool
        """
        prop = BaseFigure._str_to_dict_path(prop)
        if prop and prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
        obj = self
        for p in prop:
            if isinstance(p, int):
                if isinstance(obj, tuple) and 0 <= p < len(obj):
                    obj = obj[p]
                else:
                    return False
            elif hasattr(obj, '_valid_props') and p in obj._valid_props:
                obj = obj[p]
            else:
                return False
        return True

    def __setitem__(self, prop, value):
        """
        Parameters
        ----------
        prop : str
            The name of a direct child of this object

            Note: Setting nested properties using property path string or
            property path tuples is not supported.
        value
            New property value

        Returns
        -------
        None
        """
        from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator, BaseDataValidator
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if len(prop) == 0:
            raise KeyError(orig_prop)
        if prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
        if len(prop) == 1:
            prop = prop[0]
            if self._validate:
                if prop not in self._valid_props:
                    self._raise_on_invalid_property_error()(prop)
                validator = self._get_validator(prop)
                if isinstance(validator, CompoundValidator):
                    self._set_compound_prop(prop, value)
                elif isinstance(validator, (CompoundArrayValidator, BaseDataValidator)):
                    self._set_array_prop(prop, value)
                else:
                    self._set_prop(prop, value)
            else:
                self._init_props()
                if isinstance(value, BasePlotlyType):
                    value = value.to_plotly_json()
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], BasePlotlyType):
                    value = [v.to_plotly_json() if isinstance(v, BasePlotlyType) else v for v in value]
                self._props[prop] = value
                self._compound_props.pop(prop, None)
                self._compound_array_props.pop(prop, None)
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=ValueError)
            if err is not None:
                raise err
            res = self
            for p in prop[:-1]:
                res = res[p]
            res._validate = self._validate
            res[prop[-1]] = value

    def __setattr__(self, prop, value):
        """
        Parameters
        ----------
        prop : str
            The name of a direct child of this object
        value
            New property value
        Returns
        -------
        None
        """
        if prop.startswith('_') or hasattr(self, prop) or prop in self._valid_props:
            super(BasePlotlyType, self).__setattr__(prop, value)
        else:
            self._raise_on_invalid_property_error()(prop)

    def __iter__(self):
        """
        Return an iterator over the object's properties
        """
        res = list(self._valid_props)
        for prop in self._mapped_properties:
            res.append(prop)
        return iter(res)

    def __eq__(self, other):
        """
        Test for equality

        To be considered equal, `other` must have the same type as this object
        and their `to_plotly_json` representaitons must be identical.

        Parameters
        ----------
        other
            The object to compare against

        Returns
        -------
        bool
        """
        if not isinstance(other, self.__class__):
            return False
        else:
            return BasePlotlyType._vals_equal(self._props if self._props is not None else {}, other._props if other._props is not None else {})

    @staticmethod
    def _build_repr_for_class(props, class_name, parent_path_str=None):
        """
        Helper to build representation string for a class

        Parameters
        ----------
        class_name : str
            Name of the class being represented
        parent_path_str : str of None (default)
            Name of the class's parent package to display
        props : dict
            Properties to unpack into the constructor

        Returns
        -------
        str
            The representation string
        """
        from plotly.utils import ElidedPrettyPrinter
        if parent_path_str:
            class_name = parent_path_str + '.' + class_name
        if len(props) == 0:
            repr_str = class_name + '()'
        else:
            pprinter = ElidedPrettyPrinter(threshold=200, width=120)
            pprint_res = pprinter.pformat(props)
            body = '   ' + pprint_res[1:-1].replace('\n', '\n   ')
            repr_str = class_name + '({\n ' + body + '\n})'
        return repr_str

    def __repr__(self):
        """
        Customize object representation when displayed in the
        terminal/notebook
        """
        from _plotly_utils.basevalidators import LiteralValidator
        props = self._props if self._props is not None else {}
        props = {p: v for p, v in props.items() if p in self._valid_props and (not isinstance(self._get_validator(p), LiteralValidator))}
        if 'template' in props:
            props['template'] = '...'
        repr_str = BasePlotlyType._build_repr_for_class(props=props, class_name=self.__class__.__name__, parent_path_str=self._parent_path_str)
        return repr_str

    def _raise_on_invalid_property_error(self, _error_to_raise=None):
        """
        Returns a function that raises informative exception when invalid
        property names are encountered. The _error_to_raise argument allows
        specifying the exception to raise, which is ValueError if None.

        Parameters
        ----------
        args : list[str]
            List of property names that have already been determined to be
            invalid

        Raises
        ------
        ValueError by default, or _error_to_raise if not None
        """
        if _error_to_raise is None:
            _error_to_raise = ValueError

        def _ret(*args):
            invalid_props = args
            if invalid_props:
                if len(invalid_props) == 1:
                    prop_str = 'property'
                    invalid_str = repr(invalid_props[0])
                else:
                    prop_str = 'properties'
                    invalid_str = repr(invalid_props)
                module_root = 'plotly.graph_objs.'
                if self._parent_path_str:
                    full_obj_name = module_root + self._parent_path_str + '.' + self.__class__.__name__
                else:
                    full_obj_name = module_root + self.__class__.__name__
                guessed_prop = None
                if len(invalid_props) == 1:
                    try:
                        guessed_prop = find_closest_string(invalid_props[0], self._valid_props)
                    except Exception:
                        pass
                guessed_prop_suggestion = ''
                if guessed_prop is not None:
                    guessed_prop_suggestion = 'Did you mean "%s"?' % (guessed_prop,)
                raise _error_to_raise('Invalid {prop_str} specified for object of type {full_obj_name}: {invalid_str}\n\n{guessed_prop_suggestion}\n\n    Valid properties:\n{prop_descriptions}\n{guessed_prop_suggestion}\n'.format(prop_str=prop_str, full_obj_name=full_obj_name, invalid_str=invalid_str, prop_descriptions=self._prop_descriptions, guessed_prop_suggestion=guessed_prop_suggestion))
        return _ret

    def update(self, dict1=None, overwrite=False, **kwargs):
        """
        Update the properties of an object with a dict and/or with
        keyword arguments.

        This recursively updates the structure of the original
        object with the values in the input dict / keyword arguments.

        Parameters
        ----------
        dict1 : dict
            Dictionary of properties to be updated
        overwrite: bool
            If True, overwrite existing properties. If False, apply updates
            to existing properties recursively, preserving existing
            properties that are not specified in the update operation.
        kwargs :
            Keyword/value pair of properties to be updated

        Returns
        -------
        BasePlotlyType
            Updated plotly object
        """
        if self.figure:
            with self.figure.batch_update():
                BaseFigure._perform_update(self, dict1, overwrite=overwrite)
                BaseFigure._perform_update(self, kwargs, overwrite=overwrite)
        else:
            BaseFigure._perform_update(self, dict1, overwrite=overwrite)
            BaseFigure._perform_update(self, kwargs, overwrite=overwrite)
        return self

    def pop(self, key, *args):
        """
        Remove the value associated with the specified key and return it

        Parameters
        ----------
        key: str
            Property name
        dflt
            The default value to return if key was not found in object

        Returns
        -------
        value
            The removed value that was previously associated with key

        Raises
        ------
        KeyError
            If key is not in object and no dflt argument specified
        """
        if key not in self and args:
            return args[0]
        elif key in self:
            val = self[key]
            self[key] = None
            return val
        else:
            raise KeyError(key)

    @property
    def _in_batch_mode(self):
        """
        True if the object belongs to a figure that is currently in batch mode
        Returns
        -------
        bool
        """
        return self.parent and self.parent._in_batch_mode

    def _set_prop(self, prop, val):
        """
        Set the value of a simple property

        Parameters
        ----------
        prop : str
            Name of a simple (non-compound, non-array) property
        val
            The new property value

        Returns
        -------
        Any
            The coerced assigned value
        """
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        try:
            val = validator.validate_coerce(val)
        except ValueError as err:
            if self._skip_invalid:
                return
            else:
                raise err
        if val is None:
            if self._props and prop in self._props:
                if not self._in_batch_mode:
                    self._props.pop(prop)
                self._send_prop_set(prop, val)
        else:
            self._init_props()
            if prop not in self._props or not BasePlotlyType._vals_equal(self._props[prop], val):
                if not self._in_batch_mode:
                    self._props[prop] = val
                self._send_prop_set(prop, val)
        return val

    def _set_compound_prop(self, prop, val):
        """
        Set the value of a compound property

        Parameters
        ----------
        prop : str
            Name of a compound property
        val
            The new property value

        Returns
        -------
        BasePlotlyType
            The coerced assigned object
        """
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
        curr_val = self._compound_props.get(prop, None)
        if curr_val is not None:
            curr_dict_val = deepcopy(curr_val._props)
        else:
            curr_dict_val = None
        if val is not None:
            new_dict_val = deepcopy(val._props)
        else:
            new_dict_val = None
        if not self._in_batch_mode:
            if not new_dict_val:
                if self._props and prop in self._props:
                    self._props.pop(prop)
            else:
                self._init_props()
                self._props[prop] = new_dict_val
        if not BasePlotlyType._vals_equal(curr_dict_val, new_dict_val):
            self._send_prop_set(prop, new_dict_val)
        if isinstance(val, BasePlotlyType):
            val._parent = self
            val._orphan_props.clear()
        if curr_val is not None:
            if curr_dict_val is not None:
                curr_val._orphan_props.update(curr_dict_val)
            curr_val._parent = None
        self._compound_props[prop] = val
        return val

    def _set_array_prop(self, prop, val):
        """
        Set the value of a compound property

        Parameters
        ----------
        prop : str
            Name of a compound property
        val
            The new property value

        Returns
        -------
        tuple[BasePlotlyType]
            The coerced assigned object
        """
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
        curr_val = self._compound_array_props.get(prop, None)
        if curr_val is not None:
            curr_dict_vals = [deepcopy(cv._props) for cv in curr_val]
        else:
            curr_dict_vals = None
        if val is not None:
            new_dict_vals = [deepcopy(nv._props) for nv in val]
        else:
            new_dict_vals = None
        if not self._in_batch_mode:
            if not new_dict_vals:
                if self._props and prop in self._props:
                    self._props.pop(prop)
            else:
                self._init_props()
                self._props[prop] = new_dict_vals
        if not BasePlotlyType._vals_equal(curr_dict_vals, new_dict_vals):
            self._send_prop_set(prop, new_dict_vals)
        if val is not None:
            for v in val:
                v._orphan_props.clear()
                v._parent = self
        if curr_val is not None:
            for cv, cv_dict in zip(curr_val, curr_dict_vals):
                if cv_dict is not None:
                    cv._orphan_props.update(cv_dict)
                cv._parent = None
        self._compound_array_props[prop] = val
        return val

    def _send_prop_set(self, prop_path_str, val):
        """
        Notify parent that a property has been set to a new value

        Parameters
        ----------
        prop_path_str : str
            Property path string (e.g. 'foo[0].bar') of property that
            was set, relative to this object
        val
            New value for property. Either a simple value, a dict,
            or a tuple of dicts. This should *not* be a BasePlotlyType object.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _prop_set_child(self, child, prop_path_str, val):
        """
        Propagate property setting notification from child to parent

        Parameters
        ----------
        child : BasePlotlyType
            Child object
        prop_path_str : str
            Property path string (e.g. 'foo[0].bar') of property that
            was set, relative to `child`
        val
            New value for property. Either a simple value, a dict,
            or a tuple of dicts. This should *not* be a BasePlotlyType object.

        Returns
        -------
        None
        """
        child_prop_val = getattr(self, child.plotly_name)
        if isinstance(child_prop_val, (list, tuple)):
            child_ind = BaseFigure._index_is(child_prop_val, child)
            obj_path = '{child_name}.{child_ind}.{prop}'.format(child_name=child.plotly_name, child_ind=child_ind, prop=prop_path_str)
        else:
            obj_path = '{child_name}.{prop}'.format(child_name=child.plotly_name, prop=prop_path_str)
        self._send_prop_set(obj_path, val)

    def _restyle_child(self, child, prop, val):
        """
        Propagate _restyle_child to parent

        Note: This method must match the name and signature of the
        corresponding method on BaseFigure
        """
        self._prop_set_child(child, prop, val)

    def _relayout_child(self, child, prop, val):
        """
        Propagate _relayout_child to parent

        Note: This method must match the name and signature of the
        corresponding method on BaseFigure
        """
        self._prop_set_child(child, prop, val)

    def _dispatch_change_callbacks(self, changed_paths):
        """
        Execute the appropriate change callback functions given a set of
        changed property path tuples

        Parameters
        ----------
        changed_paths : set[tuple[int|str]]

        Returns
        -------
        None
        """
        for prop_path_tuples, callbacks in self._change_callbacks.items():
            common_paths = changed_paths.intersection(set(prop_path_tuples))
            if common_paths:
                callback_args = [self[cb_path] for cb_path in prop_path_tuples]
                for callback in callbacks:
                    callback(self, *callback_args)

    def on_change(self, callback, *args, **kwargs):
        """
        Register callback function to be called when certain properties or
        subproperties of this object are modified.

        Callback will be invoked whenever ANY of these properties is
        modified. Furthermore, the callback will only be invoked once even
        if multiple properties are modified during the same restyle /
        relayout / update operation.

        Parameters
        ----------
        callback : function
            Function that accepts 1 + len(`args`) parameters. First parameter
            is this object. Second through last parameters are the
            property / subpropery values referenced by args.
        args : list[str|tuple[int|str]]
            List of property references where each reference may be one of:

              1) A property name string (e.g. 'foo') for direct properties
              2) A property path string (e.g. 'foo[0].bar') for
                 subproperties
              3) A property path tuple (e.g. ('foo', 0, 'bar')) for
                 subproperties

        append : bool
            True if callback should be appended to previously registered
            callback on the same properties, False if callback should replace
            previously registered callbacks on the same properties. Defaults
            to False.

        Examples
        --------

        Register callback that prints out the range extents of the xaxis and
        yaxis whenever either either of them changes.

        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 0]))
        >>> fig.layout.on_change(
        ...   lambda obj, xrange, yrange: print("%s-%s" % (xrange, yrange)),
        ...   ('xaxis', 'range'), ('yaxis', 'range'))


        Returns
        -------
        None
        """
        if not self.figure:
            class_name = self.__class__.__name__
            msg = '\n{class_name} object is not a descendant of a Figure.\non_change callbacks are not supported in this case.\n'.format(class_name=class_name)
            raise ValueError(msg)
        if len(args) == 0:
            raise ValueError('At least one change property must be specified')
        invalid_args = [arg for arg in args if arg not in self]
        if invalid_args:
            raise ValueError('Invalid property specification(s): %s' % invalid_args)
        append = kwargs.get('append', False)
        arg_tuples = tuple([BaseFigure._str_to_dict_path(a) for a in args])
        if arg_tuples not in self._change_callbacks or not append:
            self._change_callbacks[arg_tuples] = []
        self._change_callbacks[arg_tuples].append(callback)

    def to_plotly_json(self):
        """
        Return plotly JSON representation of object as a Python dict

        Note: May include some JSON-invalid data types, use the `PlotlyJSONEncoder` util
        or the `to_json` method to encode to a string.

        Returns
        -------
        dict
        """
        return deepcopy(self._props if self._props is not None else {})

    def to_json(self, *args, **kwargs):
        """
        Convert object to a JSON string representation

        Parameters
        ----------
        validate: bool (default True)
            True if the object should be validated before being converted to
            JSON, False otherwise.

        pretty: bool (default False)
            True if JSON representation should be pretty-printed, False if
            representation should be as compact as possible.

        remove_uids: bool (default True)
            True if trace UIDs should be omitted from the JSON representation

        engine: str (default None)
            The JSON encoding engine to use. One of:
              - "json" for an encoder based on the built-in Python json module
              - "orjson" for a fast encoder the requires the orjson package
            If not specified, the default encoder is set to the current value of
            plotly.io.json.config.default_encoder.

        Returns
        -------
        str
            Representation of object as a JSON string
        """
        import plotly.io as pio
        return pio.to_json(self, *args, **kwargs)

    @staticmethod
    def _vals_equal(v1, v2):
        """
        Recursive equality function that handles nested dicts / tuples / lists
        that contain numpy arrays.

        v1
            First value to compare
        v2
            Second value to compare

        Returns
        -------
        bool
            True if v1 and v2 are equal, False otherwise
        """
        np = get_module('numpy', should_load=False)
        if np is not None and (isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray)):
            return np.array_equal(v1, v2)
        elif isinstance(v1, (list, tuple)):
            return isinstance(v2, (list, tuple)) and len(v1) == len(v2) and all((BasePlotlyType._vals_equal(e1, e2) for e1, e2 in zip(v1, v2)))
        elif isinstance(v1, dict):
            return isinstance(v2, dict) and set(v1.keys()) == set(v2.keys()) and all((BasePlotlyType._vals_equal(v1[k], v2[k]) for k in v1))
        else:
            return v1 == v2