import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
def Initializer(arg, allow_generators=False, treat_sequences_as_mappings=True, arg_not_specified=None):
    """Standardized processing of Component keyword arguments

    Component keyword arguments accept a number of possible inputs, from
    scalars to dictionaries, to functions (rules) and generators.  This
    function standardizes the processing of keyword arguments and
    returns "initializer classes" that are specialized to the specific
    data type provided.

    Parameters
    ----------
    arg:

        The argument passed to the component constructor.  This could
        be almost any type, including a scalar, dict, list, function,
        generator, or None.

    allow_generators: bool

        If False, then we will raise an exception if ``arg`` is a generator

    treat_sequences_as_mappings: bool

        If True, then if ``arg`` is a sequence, we will treat it as if
        it were a mapping (i.e., ``dict(enumerate(arg))``).  Otherwise
        sequences will be returned back as the value of the initializer.

    arg_not_specified:

        If ``arg`` is ``arg_not_specified``, then the function will
        return None (and not an InitializerBase object).

    """
    if arg is arg_not_specified:
        return None
    if arg.__class__ in initializer_map:
        return initializer_map[arg.__class__](arg)
    if arg.__class__ in sequence_types:
        if treat_sequences_as_mappings:
            return ItemInitializer(arg)
        else:
            return ConstantInitializer(arg)
    if arg.__class__ in function_types:
        if not allow_generators and inspect.isgeneratorfunction(arg):
            raise ValueError('Generator functions are not allowed')
        _args = inspect.getfullargspec(arg)
        _nargs = len(_args.args)
        if inspect.ismethod(arg) and arg.__self__ is not None:
            _nargs -= 1
        if _nargs == 1 and _args.varargs is None:
            return ScalarCallInitializer(arg, constant=not inspect.isgeneratorfunction(arg))
        else:
            return IndexedCallInitializer(arg)
    if hasattr(arg, '__len__'):
        if isinstance(arg, Mapping):
            initializer_map[arg.__class__] = ItemInitializer
        elif isinstance(arg, Sequence) and (not isinstance(arg, str)):
            sequence_types.add(arg.__class__)
        elif isinstance(arg, PyomoObject):
            if arg.is_component_type() and arg.is_indexed():
                initializer_map[arg.__class__] = ItemInitializer
            else:
                initializer_map[arg.__class__] = ConstantInitializer
        elif any((c.__name__ == 'ndarray' for c in arg.__class__.__mro__)):
            if numpy_available and isinstance(arg, numpy.ndarray):
                sequence_types.add(arg.__class__)
        elif any((c.__name__ == 'Series' for c in arg.__class__.__mro__)):
            if pandas_available and isinstance(arg, pandas.Series):
                sequence_types.add(arg.__class__)
        elif any((c.__name__ == 'DataFrame' for c in arg.__class__.__mro__)):
            if pandas_available and isinstance(arg, pandas.DataFrame):
                initializer_map[arg.__class__] = DataFrameInitializer
        else:
            initializer_map[arg.__class__] = ConstantInitializer
        return Initializer(arg, allow_generators=allow_generators, treat_sequences_as_mappings=treat_sequences_as_mappings, arg_not_specified=arg_not_specified)
    if inspect.isgenerator(arg) or hasattr(arg, 'next') or hasattr(arg, '__next__'):
        if not allow_generators:
            raise ValueError('Generators are not allowed')
        return ConstantInitializer(tuple(arg))
    if type(arg) is functools.partial:
        try:
            _args = inspect.getfullargspec(arg.func)
        except:
            return IndexedCallInitializer(arg)
        _positional_args = set(_args.args)
        for key in arg.keywords:
            _positional_args.discard(key)
        if len(_positional_args) - len(arg.args) == 1 and _args.varargs is None:
            return ScalarCallInitializer(arg)
        else:
            return IndexedCallInitializer(arg)
    if isinstance(arg, InitializerBase):
        return arg
    if isinstance(arg, PyomoObject):
        initializer_map[arg.__class__] = ConstantInitializer
        return ConstantInitializer(arg)
    if callable(arg) and (not isinstance(arg, type)):
        if inspect.isfunction(arg) or inspect.ismethod(arg):
            function_types.add(type(arg))
        else:
            arg = arg.__call__
        return Initializer(arg, allow_generators=allow_generators, treat_sequences_as_mappings=treat_sequences_as_mappings, arg_not_specified=arg_not_specified)
    initializer_map[arg.__class__] = ConstantInitializer
    return ConstantInitializer(arg)