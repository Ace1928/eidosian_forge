from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
class ListSweep(Sweep):
    """A wrapper around a list of `ParamResolver`s."""

    def __init__(self, resolver_list: Iterable[resolver.ParamResolverOrSimilarType]):
        """Creates a `Sweep` over a list of `ParamResolver`s.

        Args:
            resolver_list: The list of parameter resolvers to use in the sweep.
                All resolvers must resolve the same set of parameters.

        Raises:
            TypeError: If `resolver_list` is not a `cirq.ParamResolver` or a
                dict.
        """
        self.resolver_list: List[resolver.ParamResolver] = []
        for r in resolver_list:
            if not isinstance(r, (dict, resolver.ParamResolver)):
                raise TypeError(f'Not a ParamResolver or dict: <{r!r}>')
            self.resolver_list.append(resolver.ParamResolver(r))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.resolver_list == other.resolver_list

    def __ne__(self, other):
        return not self == other

    @property
    def keys(self) -> List['cirq.TParamKey']:
        if not self.resolver_list:
            return []
        return list(map(str, self.resolver_list[0].param_dict))

    def __len__(self) -> int:
        return len(self.resolver_list)

    def param_tuples(self) -> Iterator[Params]:
        for r in self.resolver_list:
            yield tuple(_params_without_symbols(r))

    def __repr__(self) -> str:
        return f'cirq.ListSweep({self.resolver_list!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['resolver_list'])