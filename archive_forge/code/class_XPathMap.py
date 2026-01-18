import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
class XPathMap(XPathFunction):
    """
    A token for processing XPath 3.1+ maps. Map instances have the double role of
    tokens and of dictionaries, depending on the way that are created (using a map
    constructor or a function). The map is fully set after the protected attribute
    _map is evaluated from tokens or initialized from arguments.
    """
    symbol = 'map'
    label = 'map'
    pattern = '(?<!\\$)\\bmap(?=\\s*(?:\\(\\:.*\\:\\))?\\s*\\{(?!\\:))'
    _map: Optional[Dict[Optional[AnyAtomicType], Any]] = None
    _values: List[XPathToken]
    _nan_key: Optional[float] = None

    def __init__(self, parser: XPathParserType, items: Optional[Any]=None) -> None:
        super().__init__(parser)
        self._values = []
        if items is not None:
            _items = items.items() if isinstance(items, dict) else items
            _map: Dict[Any, Any] = {}
            for k, v in _items:
                if k is None:
                    raise self.error('XPTY0004', 'missing key value')
                elif isinstance(k, float) and math.isnan(k):
                    if self._nan_key is not None:
                        raise self.error('XQDY0137')
                    self._nan_key, _map[None] = (k, v)
                    continue
                elif k in _map:
                    raise self.error('XQDY0137')
                if isinstance(v, list):
                    _map[k] = v[0] if len(v) == 1 else v
                else:
                    _map[k] = v
            self._map = _map

    def __repr__(self) -> str:
        if self._map is not None:
            return f'<{self.__class__.__name__} object at {hex(id(self))}>'
        return '<{} object (not evaluated constructor) at {}>'.format(self.__class__.__name__, hex(id(self)))

    def __str__(self) -> str:
        if self._map is None:
            return f'not evaluated map constructor with {len(self._items)} entries'
        return f'map{self._map}'

    def __len__(self) -> int:
        if self._map is None:
            return len(self._items)
        return len(self._map)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, XPathMap):
            if self._map is None or other._map is None:
                raise ElementPathValueError('cannot compare not evaluated maps')
            return self._map == other._map
        return NotImplemented

    def nud(self) -> 'XPathMap':
        self.parser.advance('{')
        del self._items[:]
        if self.parser.next_token.symbol not in ('}', '(end)'):
            while True:
                key = self.parser.expression(5)
                self._items.append(key)
                if self.parser.token.symbol != ':':
                    self.parser.advance(':')
                self._values.append(self.parser.expression(5))
                if self.parser.next_token.symbol != ',':
                    break
                self.parser.advance()
        self.parser.advance('}')
        return self

    @property
    def source(self) -> str:
        if self._map is None:
            items = ', '.join((f'{tk.source}:{tv.source}' for tk, tv in zip(self, self._values)))
        else:
            items = ', '.join((f'{k!r}:{v!r}' for k, v in self._map.items()))
        return f'map{{{items}}}'

    def evaluate(self, context: ContextArgType=None) -> 'XPathMap':
        if self._map is not None:
            return self
        return XPathMap(parser=self.parser, items=((k.get_atomized_operand(context), v.evaluate(context)) for k, v in zip(self._items, self._values)))

    def _evaluate(self, context: ContextArgType=None) -> Dict[AnyAtomicType, Any]:
        _map: Dict[Any, Any] = {}
        nan_key = None
        for key, value in zip(self._items, self._values):
            k = key.get_atomized_operand(context)
            if k is None:
                raise self.error('XPTY0004', 'missing key value')
            elif isinstance(k, float) and math.isnan(k):
                if nan_key is not None:
                    raise self.error('XQDY0137')
                nan_key, _map[None] = (k, value.evaluate(context))
                continue
            elif k in _map:
                raise self.error('XQDY0137')
            v = value.evaluate(context)
            if isinstance(v, list):
                _map[k] = v[0] if len(v) == 1 else v
            else:
                _map[k] = v
        self._nan_key = nan_key
        return cast(Dict[AnyAtomicType, Any], _map)

    def __call__(self, *args: XPathFunctionArgType, context: ContextArgType=None) -> Any:
        if len(args) == 1 and isinstance(args[0], list) and (len(args[0]) == 1):
            args = (args[0][0],)
        if len(args) != 1 or not isinstance(args[0], AnyAtomicType):
            if isinstance(context, XPathSchemaContext):
                return None
            raise self.error('XPST0003', 'exactly one atomic argument is expected')
        map_dict: Dict[Any, Any]
        key = args[0]
        if self._map is not None:
            map_dict = self._map
        else:
            map_dict = self._evaluate(context)
        try:
            if isinstance(key, float) and math.isnan(key):
                return map_dict[None]
            else:
                return map_dict[key]
        except KeyError:
            return []

    def keys(self, context: ContextArgType=None) -> List[AnyAtomicType]:
        if self._map is not None:
            keys = [self._nan_key if k is None else k for k in self._map.keys()]
        else:
            keys = [self._nan_key if k is None else k for k in self._evaluate(context).keys()]
        return cast(List[AnyAtomicType], keys)

    def values(self, context: ContextArgType=None) -> List[Any]:
        if self._map is not None:
            return [v for v in self._map.values()]
        return [v for v in self._evaluate(context).values()]

    def items(self, context: ContextArgType=None) -> List[Tuple[AnyAtomicType, Any]]:
        _map: Dict[Any, Any]
        if self._map is not None:
            _map = self._map
        else:
            _map = self._evaluate(context)
        return [(self._nan_key, v) if k is None else (k, v) for k, v in _map.items()]

    def match_function_test(self, function_test: Union[str, List[str]], as_argument: bool=False) -> bool:
        if isinstance(function_test, list):
            sequence_types = function_test
        else:
            sequence_types = split_function_test(function_test)
        if not sequence_types or not sequence_types[-1]:
            return False
        elif sequence_types[0] == '*':
            return True
        elif len(sequence_types) != 2:
            return False
        key_st, value_st = sequence_types
        if key_st.endswith(('+', '*')):
            return False
        elif value_st != 'empty-sequence()' and (not value_st.endswith(('?', '*'))):
            return False
        else:
            return any((match_sequence_type(k, key_st, self.parser, False) and match_sequence_type(v, value_st, self.parser) for k, v in self.items()))