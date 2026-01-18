import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
class _InlineFunction(XPathFunction):
    symbol = lookup_name = 'function'
    lbp = 90
    rbp = 90
    label = MultiLabel('inline function', 'function test')
    body: Optional[XPathToken] = None
    'Body of anonymous inline function.'
    variables: Optional[Dict[str, Any]] = None
    'In-scope variables linked by let and for expressions and arguments.'
    varnames: Optional[List[str]] = None
    'Inline function arguments varnames.'

    def __str__(self) -> str:
        return str(self.label)

    @property
    def source(self) -> str:
        if self.label == 'function test':
            if len(self.sequence_types) == 1 and self.sequence_types[0] == '*':
                return 'function(*)'
            else:
                return 'function(%s) as %s' % (', '.join(self.sequence_types[:-1]), self.sequence_types[-1])
        arguments = []
        return_type = ''
        for var, sq in zip_longest(self, self.sequence_types):
            if var is None:
                if sq != 'item()*':
                    return_type = f' as {sq}'
            elif sq is None or sq == 'item()*':
                arguments.append(var.source)
            else:
                arguments.append(f'{var.source} as {sq}')
        return '%s(%s)%s {%s}' % (self.symbol, ', '.join(arguments), return_type, getattr(self.body, 'source', ''))

    def __call__(self, *args: XPathFunctionArgType, context: Optional[XPathContext]=None) -> Any:

        def get_argument(v: Any) -> Any:
            if isinstance(v, XPathToken) and (not isinstance(v, XPathFunction)):
                v = v.evaluate(context)
            if isinstance(v, XPathFunction) and sequence_type.startswith('function('):
                if not v.match_function_test(sequence_type, as_argument=True):
                    msg = 'argument {!r}: {} does not match sequence type {}'
                    raise self.error('XPTY0004', msg.format(varname, v, sequence_type))
            elif not match_sequence_type(v, sequence_type, self.parser):
                _v = self.cast_to_primitive_type(v, sequence_type)
                if not match_sequence_type(_v, sequence_type, self.parser):
                    msg = "argument '${}': {} does not match sequence type {}"
                    raise self.error('XPTY0004', msg.format(varname, v, sequence_type))
                return _v
            return v
        self.check_arguments_number(len(args))
        context = copy(context)
        if self.variables and context is not None:
            context.variables.update(self.variables)
        if self.label == 'inline partial function':
            k = 0
            for varname, sequence_type, tk in zip(self.varnames, self.sequence_types, self):
                if tk.symbol != '?' or tk:
                    context.variables[varname] = tk.evaluate(context)
                else:
                    context.variables[varname] = get_argument(args[k])
                    k += 1
            result = self.body.evaluate(context)
        else:
            if context is None:
                raise self.missing_context()
            elif not args and self:
                if isinstance(context.item, DocumentNode):
                    if isinstance(context.root, DocumentNode):
                        context.item = context.root.getroot()
                    else:
                        context.item = context.root
                args = cast(Tuple[XPathFunctionArgType], (context.item,))
            partial_function = False
            if self.variables is None:
                self.variables = {}
            for varname, sequence_type, value in zip(self.varnames, self.sequence_types, args):
                if isinstance(value, XPathToken) and value.symbol == '?':
                    partial_function = True
                else:
                    context.variables[varname] = get_argument(value)
            if partial_function:
                self.to_partial_function()
                return self
            result = self.body.evaluate(context)
        return self.validated_result(result)

    def nud(self):

        def append_sequence_type(tk):
            if tk.symbol == '(' and len(tk) == 1:
                tk = tk[0]
            sequence_type = tk.source
            next_symbol = self.parser.next_token.symbol
            if sequence_type != 'empty-sequence()' and next_symbol in OCCURRENCE_INDICATORS:
                self.parser.advance()
                sequence_type += next_symbol
                tk.occurrence = next_symbol
            if not is_sequence_type(sequence_type, self.parser):
                if 'xs:NMTOKENS' in sequence_type or 'xs:ENTITIES' in sequence_type or 'xs:IDREFS' in sequence_type:
                    msg = 'a list type cannot be used in a function signature'
                    raise self.error('XPST0051', msg)
                raise self.error('XPST0003', 'a sequence type expected')
            self.sequence_types.append(sequence_type)
        if self.parser.next_token.symbol != '(':
            return self.as_name()
        self.parser.advance('(')
        self.sequence_types = []
        if self.parser.next_token.symbol in ('$', ')'):
            self.label = 'inline function'
            self.varnames = []
            while self.parser.next_token.symbol != ')':
                self.parser.next_token.expected('$')
                variable = self.parser.expression(5)
                varname = variable[0].value
                if varname in self.varnames:
                    raise self.error('XQST0039')
                self.append(variable)
                self.varnames.append(varname)
                if self.parser.next_token.symbol == 'as':
                    self.parser.advance('as')
                    token = self.parser.expression(90)
                    append_sequence_type(token)
                else:
                    self.sequence_types.append('item()*')
                self.parser.next_token.expected(')', ',')
                if self.parser.next_token.symbol == ',':
                    self.parser.advance()
                    self.parser.next_token.unexpected(')')
            self.parser.advance(')')
        elif self.parser.next_token.symbol == '*':
            self.label = 'function test'
            self.append(self.parser.advance('*'))
            self.sequence_types.append('*')
            self.parser.advance(')')
            return self
        else:
            self.label = 'function test'
            while True:
                token = self.parse_sequence_type()
                append_sequence_type(token)
                self.append(token)
                if self.parser.next_token.symbol != ',':
                    break
                self.parser.advance(',')
            self.parser.advance(')')
        if self.parser.next_token.symbol != 'as':
            self.sequence_types.append('item()*')
        else:
            self.parser.advance('as')
            if self.parser.next_token.label not in ('kind test', 'sequence type', 'function test'):
                self.parser.expected_next('(name)', ':')
            token = self.parser.expression(rbp=90)
            append_sequence_type(token)
        if self.label == 'inline function':
            if self.parser.next_token.symbol != '{' and (not self):
                self.label = 'function test'
            else:
                self.parser.advance('{')
                if self.parser.next_token.symbol != '}':
                    self.body = self.parser.expression()
                elif self.parser.version >= '3.1':
                    self.body = ValueToken(self.parser, value=[])
                else:
                    raise self.wrong_syntax('inline function has an empty body')
                self.parser.advance('}')
        return self

    def evaluate(self, context=None):
        if context is None:
            raise self.missing_context()
        elif self.label.endswith('function'):
            self.variables = context.variables.copy()
            return self
        if not isinstance(context.item, XPathFunction):
            return []
        elif self.source == 'function(*)':
            return context.item
        elif context.item.match_function_test(self.sequence_types):
            return context.item
        else:
            return []

    def to_partial_function(self) -> None:
        assert self.label != 'function test', 'an effective inline function required'
        nargs = len([tk and (not tk) for tk in self._items if tk.symbol == '?'])
        assert nargs, 'a partial function requires at least a placeholder token'
        self._name = None
        self.label = 'inline partial function'
        self.nargs = nargs