import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def _generic_validator_basic(validator: AnyCallable, sig: 'Signature', args: Set[str]) -> 'ValidatorCallable':
    has_kwargs = False
    if 'kwargs' in args:
        has_kwargs = True
        args -= {'kwargs'}
    if not args.issubset(all_kwargs):
        raise ConfigError(f'Invalid signature for validator {validator}: {sig}, should be: (value, values, config, field), "values", "config" and "field" are all optional.')
    if has_kwargs:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field, config=config)
    elif args == set():
        return lambda cls, v, values, field, config: validator(v)
    elif args == {'values'}:
        return lambda cls, v, values, field, config: validator(v, values=values)
    elif args == {'field'}:
        return lambda cls, v, values, field, config: validator(v, field=field)
    elif args == {'config'}:
        return lambda cls, v, values, field, config: validator(v, config=config)
    elif args == {'values', 'field'}:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field)
    elif args == {'values', 'config'}:
        return lambda cls, v, values, field, config: validator(v, values=values, config=config)
    elif args == {'field', 'config'}:
        return lambda cls, v, values, field, config: validator(v, field=field, config=config)
    else:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field, config=config)