from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
class CacheKey(NamedTuple):
    """The key used to identify a SQL statement construct in the
    SQL compilation cache.

    .. seealso::

        :ref:`sql_caching`

    """
    key: Tuple[Any, ...]
    bindparams: Sequence[BindParameter[Any]]

    def __hash__(self) -> Optional[int]:
        """CacheKey itself is not hashable - hash the .key portion"""
        return None

    def to_offline_string(self, statement_cache: MutableMapping[Any, str], statement: ClauseElement, parameters: _CoreSingleExecuteParams) -> str:
        """Generate an "offline string" form of this :class:`.CacheKey`

        The "offline string" is basically the string SQL for the
        statement plus a repr of the bound parameter values in series.
        Whereas the :class:`.CacheKey` object is dependent on in-memory
        identities in order to work as a cache key, the "offline" version
        is suitable for a cache that will work for other processes as well.

        The given ``statement_cache`` is a dictionary-like object where the
        string form of the statement itself will be cached.  This dictionary
        should be in a longer lived scope in order to reduce the time spent
        stringifying statements.


        """
        if self.key not in statement_cache:
            statement_cache[self.key] = sql_str = str(statement)
        else:
            sql_str = statement_cache[self.key]
        if not self.bindparams:
            param_tuple = tuple((parameters[key] for key in sorted(parameters)))
        else:
            param_tuple = tuple((parameters.get(bindparam.key, bindparam.value) for bindparam in self.bindparams))
        return repr((sql_str, param_tuple))

    def __eq__(self, other: Any) -> bool:
        return bool(self.key == other.key)

    def __ne__(self, other: Any) -> bool:
        return not self.key == other.key

    @classmethod
    def _diff_tuples(cls, left: CacheKey, right: CacheKey) -> str:
        ck1 = CacheKey(left, [])
        ck2 = CacheKey(right, [])
        return ck1._diff(ck2)

    def _whats_different(self, other: CacheKey) -> Iterator[str]:
        k1 = self.key
        k2 = other.key
        stack: List[int] = []
        pickup_index = 0
        while True:
            s1, s2 = (k1, k2)
            for idx in stack:
                s1 = s1[idx]
                s2 = s2[idx]
            for idx, (e1, e2) in enumerate(zip_longest(s1, s2)):
                if idx < pickup_index:
                    continue
                if e1 != e2:
                    if isinstance(e1, tuple) and isinstance(e2, tuple):
                        stack.append(idx)
                        break
                    else:
                        yield ('key%s[%d]:  %s != %s' % (''.join(('[%d]' % id_ for id_ in stack)), idx, e1, e2))
            else:
                pickup_index = stack.pop(-1)
                break

    def _diff(self, other: CacheKey) -> str:
        return ', '.join(self._whats_different(other))

    def __str__(self) -> str:
        stack: List[Union[Tuple[Any, ...], HasCacheKey]] = [self.key]
        output = []
        sentinel = object()
        indent = -1
        while stack:
            elem = stack.pop(0)
            if elem is sentinel:
                output.append(' ' * (indent * 2) + '),')
                indent -= 1
            elif isinstance(elem, tuple):
                if not elem:
                    output.append(' ' * ((indent + 1) * 2) + '()')
                else:
                    indent += 1
                    stack = list(elem) + [sentinel] + stack
                    output.append(' ' * (indent * 2) + '(')
            else:
                if isinstance(elem, HasCacheKey):
                    repr_ = '<%s object at %s>' % (type(elem).__name__, hex(id(elem)))
                else:
                    repr_ = repr(elem)
                output.append(' ' * (indent * 2) + '  ' + repr_ + ', ')
        return 'CacheKey(key=%s)' % ('\n'.join(output),)

    def _generate_param_dict(self) -> Dict[str, Any]:
        """used for testing"""
        _anon_map = prefix_anon_map()
        return {b.key % _anon_map: b.effective_value for b in self.bindparams}

    @util.preload_module('sqlalchemy.sql.elements')
    def _apply_params_to_element(self, original_cache_key: CacheKey, target_element: ColumnElement[Any]) -> ColumnElement[Any]:
        if target_element._is_immutable or original_cache_key is self:
            return target_element
        elements = util.preloaded.sql_elements
        return elements._OverrideBinds(target_element, self.bindparams, original_cache_key.bindparams)