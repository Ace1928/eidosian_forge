import numbers
from typing import Any, cast, Dict, Iterator, Mapping, Optional, TYPE_CHECKING, Union
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from cirq._compat import proper_repr
from cirq._doc import document
class ParamResolver:
    """Resolves parameters to actual values.

    A parameter is a variable whose value has not been determined.
    A ParamResolver is an object that can be used to assign values for these
    variables.

    ParamResolvers are hashable; their param_dict must not be mutated.

    Attributes:
        param_dict: A dictionary from the ParameterValue key (str) to its
            assigned value.

    Raises:
        TypeError if formulas are passed as keys.
    """

    def __new__(cls, param_dict: 'cirq.ParamResolverOrSimilarType'=None):
        if isinstance(param_dict, ParamResolver):
            return param_dict
        return super().__new__(cls)

    def __init__(self, param_dict: 'cirq.ParamResolverOrSimilarType'=None) -> None:
        if hasattr(self, 'param_dict'):
            return
        self._param_hash: Optional[int] = None
        self._param_dict = cast(ParamDictType, {} if param_dict is None else param_dict)
        for key in self._param_dict:
            if isinstance(key, sympy.Expr) and (not isinstance(key, sympy.Symbol)):
                raise TypeError(f'ParamResolver keys cannot be (non-symbol) formulas ({key})')
        self._deep_eval_map: ParamDictType = {}

    @property
    def param_dict(self) -> ParamMappingType:
        return self._param_dict

    def value_of(self, value: Union['cirq.TParamKey', 'cirq.TParamValComplex'], recursive: bool=True) -> 'cirq.TParamValComplex':
        """Attempt to resolve a parameter to its assigned value.

        Scalars are returned without modification.  Strings are resolved via
        the parameter dictionary with exact match only.  Otherwise, strings
        are considered to be sympy.Symbols with the name as the input string.

        A sympy.Symbol is first checked for exact match in the parameter
        dictionary. Otherwise, it is treated as a sympy.Basic.

        A sympy.Basic is resolved using sympy substitution.

        Note that passing a formula to this resolver can be slow due to the
        underlying sympy library.  For circuits relying on quick performance,
        it is recommended that all formulas are flattened before-hand using
        cirq.flatten or other means so that formula resolution is avoided.
        If unable to resolve a sympy.Symbol, returns it unchanged.
        If unable to resolve a name, returns a sympy.Symbol with that name.

        Args:
            value: The parameter to try to resolve.
            recursive: Whether to recursively evaluate formulas.

        Returns:
            The value of the parameter as resolved by this resolver.

        Raises:
            RecursionError: If the ParamResolver detects a loop in recursive
                resolution.
            sympy.SympifyError: If the resulting value cannot be interpreted.
        """
        v = _resolve_value(value)
        if v is not NotImplemented:
            return v
        if isinstance(value, (str, sympy.Symbol)):
            string = value if isinstance(value, str) else value.name
            symbol = value if isinstance(value, sympy.Symbol) else sympy.Symbol(value)
            param_value = self._param_dict.get(string, _NOT_FOUND)
            if param_value is _NOT_FOUND:
                param_value = self._param_dict.get(symbol, _NOT_FOUND)
            if param_value is _NOT_FOUND:
                return symbol
            v = _resolve_value(param_value)
            if v is not NotImplemented:
                return v
            if isinstance(param_value, str):
                param_value = sympy.Symbol(param_value)
            elif not isinstance(param_value, sympy.Basic):
                return value
            if recursive:
                param_value = self._value_of_recursive(value)
            return param_value
        if not isinstance(value, sympy.Basic):
            return value
        if isinstance(value, sympy.Float):
            return float(value)
        if isinstance(value, sympy.Add):
            summation = self.value_of(value.args[0], recursive)
            for addend in value.args[1:]:
                summation += self.value_of(addend, recursive)
            return summation
        if isinstance(value, sympy.Mul):
            product = self.value_of(value.args[0], recursive)
            for factor in value.args[1:]:
                product *= self.value_of(factor, recursive)
            return product
        if isinstance(value, sympy.Pow) and len(value.args) == 2:
            base = self.value_of(value.args[0], recursive)
            exponent = self.value_of(value.args[1], recursive)
            if isinstance(base, numbers.Number):
                return np.float_power(cast(complex, base), cast(complex, exponent))
            return np.power(cast(complex, base), cast(complex, exponent))
        if not recursive:
            v = value.subs(self._param_dict, simultaneous=True)
            if v.free_symbols:
                return v
            elif sympy.im(v):
                return complex(v)
            else:
                return float(v)
        return self._value_of_recursive(value)

    def _value_of_recursive(self, value: 'cirq.TParamKey') -> 'cirq.TParamValComplex':
        if value in self._deep_eval_map:
            v = self._deep_eval_map[value]
            if v is _RECURSION_FLAG:
                raise RecursionError('Evaluation of {value} indirectly contains itself.')
            return v
        self._deep_eval_map[value] = _RECURSION_FLAG
        v = self.value_of(value, recursive=False)
        if v == value:
            self._deep_eval_map[value] = v
        else:
            self._deep_eval_map[value] = self.value_of(v, recursive=True)
        return self._deep_eval_map[value]

    def _resolve_parameters_(self, resolver: 'ParamResolver', recursive: bool) -> 'ParamResolver':
        new_dict: Dict['cirq.TParamKey', Union[float, str, sympy.Symbol, sympy.Expr]] = {k: k for k in resolver}
        new_dict.update({k: self.value_of(k, recursive) for k in self})
        new_dict.update({k: resolver.value_of(v, recursive) for k, v in new_dict.items()})
        if recursive and self._param_dict:
            new_resolver = ParamResolver(cast(ParamDictType, new_dict))
            return ParamResolver()._resolve_parameters_(new_resolver, recursive=True)
        return ParamResolver(cast(ParamDictType, new_dict))

    def __iter__(self) -> Iterator[Union[str, sympy.Expr]]:
        return iter(self._param_dict)

    def __bool__(self) -> bool:
        return bool(self._param_dict)

    def __getitem__(self, key: Union['cirq.TParamKey', 'cirq.TParamValComplex']) -> 'cirq.TParamValComplex':
        return self.value_of(key)

    def __hash__(self) -> int:
        if self._param_hash is None:
            self._param_hash = hash(frozenset(self._param_dict.items()))
        return self._param_hash

    def __eq__(self, other):
        if not isinstance(other, ParamResolver):
            return NotImplemented
        return self._param_dict == other._param_dict

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        param_dict_repr = '{' + ', '.join((f'{proper_repr(k)}: {proper_repr(v)}' for k, v in self._param_dict.items())) + '}'
        return f'cirq.ParamResolver({param_dict_repr})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {'param_dict': list(self._param_dict.items())}

    @classmethod
    def _from_json_dict_(cls, param_dict, **kwargs):
        return cls(dict(param_dict))