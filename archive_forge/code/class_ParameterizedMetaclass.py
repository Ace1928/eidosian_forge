import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
class ParameterizedMetaclass(type):
    """
    The metaclass of Parameterized (and all its descendents).

    The metaclass overrides type.__setattr__ to allow us to set
    Parameter values on classes without overwriting the attribute
    descriptor.  That is, for a Parameterized class of type X with a
    Parameter y, the user can type X.y=3, which sets the default value
    of Parameter y to be 3, rather than overwriting y with the
    constant value 3 (and thereby losing all other info about that
    Parameter, such as the doc string, bounds, etc.).

    The __init__ method is used when defining a Parameterized class,
    usually when the module where that class is located is imported
    for the first time.  That is, the __init__ in this metaclass
    initializes the *class* object, while the __init__ method defined
    in each Parameterized class is called for each new instance of
    that class.

    Additionally, a class can declare itself abstract by having an
    attribute __abstract set to True. The 'abstract' attribute can be
    used to find out if a class is abstract or not.
    """

    def __init__(mcs, name, bases, dict_):
        """
        Initialize the class object (not an instance of the class, but
        the class itself).

        Initializes all the Parameters by looking up appropriate
        default values (see __param_inheritance()) and setting
        attrib_names (see _set_names()).
        """
        type.__init__(mcs, name, bases, dict_)
        explicit_no_refs = set()
        for base in bases:
            if issubclass(base, Parameterized):
                explicit_no_refs |= set(base._param__private.explicit_no_refs)
        _param__private = _ClassPrivate(explicit_no_refs=list(explicit_no_refs))
        mcs._param__private = _param__private
        mcs.__set_name(name, dict_)
        mcs._param__parameters = Parameters(mcs)
        parameters = [(n, o) for n, o in dict_.items() if isinstance(o, Parameter)]
        for param_name, param in parameters:
            mcs._initialize_parameter(param_name, param)
        dependers = [(n, m, m._dinfo) for n, m in dict_.items() if hasattr(m, '_dinfo')]
        _watch = []
        for name, method, dinfo in dependers:
            watch = dinfo.get('watch', False)
            on_init = dinfo.get('on_init', False)
            minfo = MInfo(cls=mcs, inst=None, name=name, method=method)
            deps, dynamic_deps = _params_depended_on(minfo, dynamic=False)
            if watch:
                _watch.append((name, watch == 'queued', on_init, deps, dynamic_deps))
        _inherited = []
        for cls in classlist(mcs)[:-1][::-1]:
            if not hasattr(cls, '_param__parameters'):
                continue
            for dep in cls.param._depends['watch']:
                method = getattr(mcs, dep[0], None)
                dinfo = getattr(method, '_dinfo', {'watch': False})
                if not any((dep[0] == w[0] for w in _watch + _inherited)) and dinfo.get('watch'):
                    _inherited.append(dep)
        mcs.param._depends = {'watch': _inherited + _watch}
        if docstring_signature:
            mcs.__class_docstring()

    def __set_name(mcs, name, dict_):
        """
        Give Parameterized classes a useful 'name' attribute that is by
        default the class name, unless a class in the hierarchy has defined
        a `name` String Parameter with a defined `default` value, in which case
        that value is used to set the class name.
        """
        name_param = dict_.get('name', None)
        if name_param is not None:
            if not type(name_param) is String:
                raise TypeError(f"Parameterized class {name!r} cannot override the 'name' Parameter with type {type(name_param)}. Overriding 'name' is only allowed with a 'String' Parameter.")
            if name_param.default:
                mcs.name = name_param.default
                mcs._param__private.renamed = True
            else:
                mcs.name = name
        else:
            classes = classlist(mcs)[::-1]
            found_renamed = False
            for c in classes:
                if hasattr(c, '_param__private') and c._param__private.renamed:
                    found_renamed = True
                    break
            if not found_renamed:
                mcs.name = name

    def __class_docstring(mcs):
        """
        Customize the class docstring with a Parameter table if
        `docstring_describe_params` and the `param_pager` is available.
        """
        if not docstring_describe_params or not param_pager:
            return
        class_docstr = mcs.__doc__ if mcs.__doc__ else ''
        description = param_pager(mcs)
        mcs.__doc__ = class_docstr + '\n' + description

    def _initialize_parameter(mcs, param_name, param):
        param._set_names(param_name)
        mcs.__param_inheritance(param_name, param)

    def __is_abstract(mcs):
        """
        Return True if the class has an attribute __abstract set to True.
        Subclasses will return False unless they themselves have
        __abstract set to true.  This mechanism allows a class to
        declare itself to be abstract (e.g. to avoid it being offered
        as an option in a GUI), without the "abstract" property being
        inherited by its subclasses (at least one of which is
        presumably not abstract).
        """
        try:
            return getattr(mcs, '_%s__abstract' % mcs.__name__.lstrip('_'))
        except AttributeError:
            return False

    def __get_signature(mcs):
        """
        For classes with a constructor signature that matches the default
        Parameterized.__init__ signature (i.e. ``__init__(self, **params)``)
        this method will generate a new signature that expands the
        parameters. If the signature differs from the default the
        custom signature is returned.
        """
        if mcs._param__private.signature:
            return mcs._param__private.signature
        if inspect.signature(mcs.__init__) != DEFAULT_SIGNATURE:
            return None
        processed_kws, keyword_groups = (set(), [])
        for cls in reversed(mcs.mro()):
            keyword_group = []
            for k, v in sorted(cls.__dict__.items()):
                if isinstance(v, Parameter) and k not in processed_kws and (not v.readonly):
                    keyword_group.append(k)
                    processed_kws.add(k)
            keyword_groups.append(keyword_group)
        keywords = [el for grp in reversed(keyword_groups) for el in grp]
        mcs._param__private.signature = signature = inspect.Signature([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in keywords])
        return signature
    __signature__ = property(__get_signature)
    abstract = property(__is_abstract)

    def _get_param(mcs):
        return mcs._param__parameters
    param = property(_get_param)

    def __setattr__(mcs, attribute_name, value):
        """
        Implements 'self.attribute_name=value' in a way that also supports Parameters.

        If there is already a descriptor named attribute_name, and
        that descriptor is a Parameter, and the new value is *not* a
        Parameter, then call that Parameter's __set__ method with the
        specified value.

        In all other cases set the attribute normally (i.e. overwrite
        the descriptor).  If the new value is a Parameter, once it has
        been set we make sure that the value is inherited from
        Parameterized superclasses as described in __param_inheritance().
        """
        parameter, owning_class = mcs.get_param_descriptor(attribute_name)
        if parameter and (not isinstance(value, Parameter)):
            if owning_class != mcs:
                parameter = copy.copy(parameter)
                parameter.owner = mcs
                type.__setattr__(mcs, attribute_name, parameter)
            mcs.__dict__[attribute_name].__set__(None, value)
        else:
            type.__setattr__(mcs, attribute_name, value)
            if isinstance(value, Parameter):
                mcs.__param_inheritance(attribute_name, value)

    def __param_inheritance(mcs, param_name, param):
        """
        Look for Parameter values in superclasses of this
        Parameterized class.

        Ordinarily, when a Python object is instantiated, attributes
        not given values in the constructor will inherit the value
        given in the object's class, or in its superclasses.  For
        Parameters owned by Parameterized classes, we have implemented
        an additional level of default lookup, should this ordinary
        lookup return only `Undefined`.

        In such a case, i.e. when no non-`Undefined` value was found for a
        Parameter by the usual inheritance mechanisms, we explicitly
        look for Parameters with the same name in superclasses of this
        Parameterized class, and use the first such value that we
        find.

        The goal is to be able to set the default value (or other
        slots) of a Parameter within a Parameterized class, just as we
        can set values for non-Parameter objects in Parameterized
        classes, and have the values inherited through the
        Parameterized hierarchy as usual.

        Note that instantiate is handled differently: if there is a
        parameter with the same name in one of the superclasses with
        instantiate set to True, this parameter will inherit
        instantiate=True.
        """
        p_type = type(param)
        slots = dict.fromkeys(p_type._all_slots_)
        setattr(param, 'owner', mcs)
        del slots['owner']
        if 'objtype' in slots:
            setattr(param, 'objtype', mcs)
            del slots['objtype']
        supers = classlist(mcs)[::-1]
        type_change = False
        for superclass in supers:
            super_param = superclass.__dict__.get(param_name)
            if not isinstance(super_param, Parameter):
                continue
            if super_param.instantiate is True:
                param.instantiate = True
            super_type = type(super_param)
            if not issubclass(super_type, p_type):
                type_change = True
        del slots['instantiate']
        callables, slot_values = ({}, {})
        slot_overridden = False
        for slot in slots.keys():
            for scls in supers:
                new_param = scls.__dict__.get(param_name)
                if new_param is None or not hasattr(new_param, slot):
                    continue
                new_value = getattr(new_param, slot)
                old_value = slot_values.get(slot, Undefined)
                if new_value is Undefined:
                    continue
                elif new_value is old_value:
                    continue
                elif old_value is Undefined:
                    slot_values[slot] = new_value
                    if slot_overridden or type_change:
                        break
                else:
                    if slot not in param._non_validated_slots:
                        slot_overridden = True
                    break
            if slot_values.get(slot, Undefined) is Undefined:
                try:
                    default_val = param._slot_defaults[slot]
                except KeyError as e:
                    raise KeyError(f'Slot {slot!r} of parameter {param_name!r} has no default value defined in `_slot_defaults`') from e
                if callable(default_val):
                    callables[slot] = default_val
                else:
                    slot_values[slot] = default_val
            elif slot == 'allow_refs':
                explicit_no_refs = mcs._param__private.explicit_no_refs
                if param.allow_refs is False:
                    explicit_no_refs.append(param.name)
                elif param.allow_refs is True and param.name in explicit_no_refs:
                    explicit_no_refs.remove(param.name)
        for slot, value in slot_values.items():
            setattr(param, slot, value)
            if slot != 'default':
                v = getattr(param, slot)
                if _is_mutable_container(v):
                    setattr(param, slot, copy.copy(v))
        for slot, fn in callables.items():
            setattr(param, slot, fn(param))
        param._update_state()
        if type_change or (slot_overridden and param.default is not None):
            try:
                param._validate(param.default)
            except Exception as e:
                msg = f'{_validate_error_prefix(param)} failed to validate its default value on class creation, this is going to raise an error in the future. '
                parents = ', '.join((klass.__name__ for klass in mcs.__mro__[1:-2]))
                if not type_change and slot_overridden:
                    msg += f'The Parameter is defined with attributes which when combined with attributes inherited from its parent classes ({parents}) make it invalid. Please fix the Parameter attributes.'
                elif type_change and (not slot_overridden):
                    msg += f'The Parameter type changed between class {mcs.__name__!r} and one of its parent classes ({parents}) which made it invalid. Please fix the Parameter type.'
                else:
                    pass
                msg += f'\nValidation failed with:\n{e}'
                warnings.warn(msg, category=_ParamFutureWarning, stacklevel=4)

    def get_param_descriptor(mcs, param_name):
        """
        Goes up the class hierarchy (starting from the current class)
        looking for a Parameter class attribute param_name. As soon as
        one is found as a class attribute, that Parameter is returned
        along with the class in which it is declared.
        """
        classes = classlist(mcs)
        for c in classes[::-1]:
            attribute = c.__dict__.get(param_name)
            if isinstance(attribute, Parameter):
                return (attribute, c)
        return (None, None)