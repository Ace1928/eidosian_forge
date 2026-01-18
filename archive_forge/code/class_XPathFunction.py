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
class XPathFunction(XPathToken):
    """
    A token for processing XPath functions.
    """
    __name__: str
    _name: Optional[QName] = None
    pattern = '(?<!\\$)\\b[^\\d\\W][\\w.\\-\\xb7\\u0300-\\u036F\\u203F\\u2040]*(?=\\s*(?:\\(\\:.*\\:\\))?\\s*\\((?!\\:))'
    sequence_types: Union[List[str], Tuple[str, ...]] = ()
    'Sequence types of arguments and of the return value of the function.'
    nargs: NargsType = None
    'Number of arguments: a single value or a couple with None that means unbounded.'
    context: ContextArgType = None
    'Dynamic context associated by function reference evaluation or explicitly by a builder.'

    def __init__(self, parser: XPathParserType, nargs: Optional[int]=None) -> None:
        super().__init__(parser)
        if isinstance(nargs, int) and nargs != self.nargs:
            if nargs < 0:
                raise self.error('XPST0017', 'number of arguments must be non negative')
            elif self.nargs is None:
                self.nargs = nargs
            elif isinstance(self.nargs, int):
                raise self.error('XPST0017', 'incongruent number of arguments')
            elif self.nargs[0] > nargs or (self.nargs[1] is not None and self.nargs[1] < nargs):
                raise self.error('XPST0017', 'incongruent number of arguments')
            else:
                self.nargs = nargs

    def __repr__(self) -> str:
        qname = self.name
        if qname is None:
            return '<%s object at %#x>' % (self.__class__.__name__, id(self))
        elif not isinstance(self.nargs, int):
            return '<XPathFunction %s at %#x>' % (qname.qname, id(self))
        return '<XPathFunction %s#%r at %#x>' % (qname.qname, self.nargs, id(self))

    def __str__(self) -> str:
        if self.namespace is None:
            return f'{self.symbol!r} {self.label}'
        elif self.namespace == XPATH_FUNCTIONS_NAMESPACE:
            return f"'fn:{self.symbol}' {self.label}"
        else:
            for prefix, uri in self.parser.namespaces.items():
                if uri == self.namespace:
                    return f"'{prefix}:{self.symbol}' {self.label}"
            else:
                return f"'Q{{{self.namespace}}}{self.symbol}' {self.label}"

    def __call__(self, *args: XPathFunctionArgType, context: ContextArgType=None) -> Any:
        self.check_arguments_number(len(args))
        context = copy(self.context or context)
        if self.label == 'partial function':
            for value, tk in zip(args, filter(lambda x: x.symbol == '?', self)):
                if isinstance(value, XPathToken) and (not isinstance(value, XPathFunction)):
                    tk.value = value.evaluate(context)
                else:
                    tk.value = value
        else:
            self.clear()
            for value in args:
                if isinstance(value, XPathToken):
                    self._items.append(value)
                elif value is None or isinstance(value, (XPathNode, AnyAtomicType, list)):
                    self._items.append(ValueToken(self.parser, value=value))
                elif not is_etree_document(value) and (not is_etree_element(value)):
                    raise self.error('XPTY0004', f'unexpected argument type {type(value)}')
                else:
                    if context is not None:
                        value = context.get_context_item(cast(Union[ElementProtocol, DocumentProtocol], value), namespaces=context.namespaces, uri=self.parser.base_uri)
                    else:
                        value = get_node_tree(cast(Union[ElementProtocol, DocumentProtocol], value), namespaces=self.parser.namespaces, uri=self.parser.base_uri)
                    self._items.append(ValueToken(self.parser, value=value))
            if any((tk.symbol == '?' and (not tk) for tk in self._items)):
                self.to_partial_function()
                return self
        if isinstance(self.label, MultiLabel):
            if self.namespace == XSD_NAMESPACE and 'constructor function' in self.label.values:
                self.label = 'constructor function'
            else:
                for label in self.label.values:
                    if label.endswith('function'):
                        self.label = label
                        break
        if self.label == 'partial function':
            result = self._partial_evaluate(context)
        else:
            result = self.evaluate(context)
        return self.validated_result(result)

    def check_arguments_number(self, nargs: int) -> None:
        """Check the number of arguments against function arity."""
        if self.nargs is None or self.nargs == nargs:
            pass
        elif isinstance(self.nargs, tuple):
            if nargs < self.nargs[0]:
                raise self.error('XPTY0004', 'missing required arguments')
            elif self.nargs[1] is not None and nargs > self.nargs[1]:
                raise self.error('XPTY0004', 'too many arguments')
        elif self.nargs > nargs:
            raise self.error('XPTY0004', 'missing required arguments')
        else:
            raise self.error('XPTY0004', 'too many arguments')

    def validated_result(self, result: Any) -> Any:
        if isinstance(result, XPathToken) and result.symbol == '?':
            return result
        elif match_sequence_type(result, self.sequence_types[-1], self.parser):
            return result
        _result = self.cast_to_primitive_type(result, self.sequence_types[-1])
        if not match_sequence_type(_result, self.sequence_types[-1], self.parser):
            msg = '{!r} does not match sequence type {}'
            raise self.error('XPTY0004', msg.format(result, self.sequence_types[-1]))
        return _result

    @property
    def source(self) -> str:
        if self.label in ('sequence type', 'kind test', ''):
            return '%s(%s)%s' % (self.symbol, ', '.join((item.source for item in self)), self.occurrence or '')
        return '%s(%s)' % (self.symbol, ', '.join((item.source for item in self)))

    @property
    def name(self) -> Optional[QName]:
        if self._name is not None:
            return self._name
        elif self.symbol == 'function':
            return None
        elif self.label == 'partial function':
            return None
        elif not self.namespace:
            self._name = QName(None, self.symbol)
        elif self.namespace == XPATH_FUNCTIONS_NAMESPACE:
            self._name = QName(XPATH_FUNCTIONS_NAMESPACE, 'fn:%s' % self.symbol)
        elif self.namespace == XSD_NAMESPACE:
            self._name = QName(XSD_NAMESPACE, 'xs:%s' % self.symbol)
        elif self.namespace == XPATH_MATH_FUNCTIONS_NAMESPACE:
            self._name = QName(XPATH_MATH_FUNCTIONS_NAMESPACE, 'math:%s' % self.symbol)
        else:
            for pfx, uri in self.parser.namespaces.items():
                if uri == self.namespace:
                    self._name = QName(uri, f'{pfx}:{self.symbol}')
                    break
            else:
                self._name = QName(self.namespace, self.symbol)
        return self._name

    @property
    def arity(self) -> int:
        if isinstance(self.nargs, int):
            return self.nargs
        return len(self._items)

    @property
    def min_args(self) -> int:
        if isinstance(self.nargs, int):
            return self.nargs
        elif isinstance(self.nargs, (tuple, list)):
            return self.nargs[0]
        else:
            return 0

    @property
    def max_args(self) -> Optional[int]:
        if isinstance(self.nargs, int):
            return self.nargs
        elif isinstance(self.nargs, (tuple, list)):
            return self.nargs[1]
        else:
            return None

    def is_reference(self) -> int:
        if not isinstance(self.nargs, int):
            return False
        return self.nargs and (not len(self._items))

    def nud(self) -> 'XPathFunction':
        self.value = None
        if not self.parser.parse_arguments:
            return self
        code = 'XPST0017' if self.label == 'function' else 'XPST0003'
        self.parser.advance('(')
        if self.nargs is None:
            del self._items[:]
            if self.parser.next_token.symbol in (')', '(end)'):
                raise self.error(code, 'at least an argument is required')
            while True:
                self.append(self.parser.expression(5))
                if self.parser.next_token.symbol != ',':
                    break
                self.parser.advance()
        elif self.nargs == 0:
            if self.parser.next_token.symbol != ')':
                if self.parser.next_token.symbol != '(end)':
                    raise self.error(code, '%s has no arguments' % str(self))
                raise self.parser.next_token.wrong_syntax()
            self.parser.advance()
            return self
        else:
            if isinstance(self.nargs, (tuple, list)):
                min_args, max_args = self.nargs
            else:
                min_args = max_args = self.nargs
            k = 0
            while k < min_args:
                if self.parser.next_token.symbol in (')', '(end)'):
                    msg = 'Too few arguments: expected at least %s arguments' % min_args
                    raise self.error('XPST0017', msg if min_args > 1 else msg[:-1])
                self._items[k:] = (self.parser.expression(5),)
                k += 1
                if k < min_args:
                    if self.parser.next_token.symbol == ')':
                        msg = f'{str(self)}: Too few arguments, expected at least {min_args} arguments'
                        raise self.error(code, msg if min_args > 1 else msg[:-1])
                    self.parser.advance(',')
            while max_args is None or k < max_args:
                if self.parser.next_token.symbol == ',':
                    self.parser.advance(',')
                    self._items[k:] = (self.parser.expression(5),)
                elif k == 0 and self.parser.next_token.symbol != ')':
                    self._items[k:] = (self.parser.expression(5),)
                else:
                    break
                k += 1
            if self.parser.next_token.symbol == ',':
                msg = 'Too many arguments: expected at most %s arguments' % max_args
                raise self.error(code, msg if max_args != 1 else msg[:-1])
        self.parser.advance(')')
        if any((tk.symbol == '?' and (not tk) for tk in self._items)):
            self.to_partial_function()
        return self

    def match_function_test(self, function_test: Union[str, List[str]], as_argument: bool=False) -> bool:
        """
        Match if function signature satisfies the provided *function_test*.
        For default return type is covariant and arguments are contravariant.
        If *as_argument* is `True` the match is inverted.

        References:
          https://www.w3.org/TR/xpath-31/#id-function-test
          https://www.w3.org/TR/xpath-31/#id-sequencetype-subtype
        """
        if isinstance(function_test, list):
            sequence_types = function_test
        else:
            sequence_types = split_function_test(function_test)
        if not sequence_types or not sequence_types[-1]:
            return False
        elif sequence_types[0] == '*':
            return True
        signature = [x for x in self.sequence_types[:self.arity]]
        signature.append(self.sequence_types[-1])
        if len(sequence_types) != len(signature):
            return False
        if as_argument:
            iterator = zip(sequence_types[:-1], signature[:-1])
        else:
            iterator = zip(signature[:-1], sequence_types[:-1])
        for st1, st2 in iterator:
            if not is_sequence_type_restriction(st1, st2):
                return False
        else:
            st1, st2 = (sequence_types[-1], signature[-1])
            return is_sequence_type_restriction(st1, st2)

    def to_partial_function(self) -> None:
        """Convert an XPath function to a partial function."""
        nargs = len([tk and (not tk) for tk in self._items if tk.symbol == '?'])
        assert nargs, 'a partial function requires at least a placeholder token'
        if self.label != 'partial function':

            def evaluate(context: ContextArgType=None) -> Any:
                return self

            def select(context: ContextArgType=None) -> Any:
                yield self
            if self.__class__.evaluate is not XPathToken.evaluate:
                setattr(self, '_partial_evaluate', self.evaluate)
            if self.__class__.select is not XPathToken.select:
                setattr(self, '_partial_select', self.select)
            setattr(self, 'evaluate', evaluate)
            setattr(self, 'select', select)
        self._name = None
        self.label = 'partial function'
        self.nargs = nargs

    def as_function(self) -> Callable[..., Any]:
        """
        Wraps the XPath function instance into a standard function.
        """

        def wrapper(*args: XPathFunctionArgType, context: ContextArgType=None) -> Any:
            return self.__call__(*args, context=context)
        qname = self.name
        if self.is_reference():
            ref_part = f'#{self.nargs}'
        else:
            ref_part = ''
        if qname is None:
            name = f'<anonymous-function{ref_part}>'
        else:
            name = f'<{qname.qname}{ref_part}>'
        wrapper.__name__ = name
        wrapper.__qualname__ = wrapper.__qualname__[:-7] + name
        return wrapper

    def _partial_evaluate(self, context: ContextArgType=None) -> Any:
        return [x for x in self._partial_select(context)]

    def _partial_select(self, context: ContextArgType=None) -> Iterator[Any]:
        item = self._partial_evaluate(context)
        if item is not None:
            if isinstance(item, list):
                yield from item
            else:
                if context is not None:
                    context.item = item
                yield item