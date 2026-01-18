import math
import decimal
import operator
from copy import copy
from ..datatypes import AnyURI
from ..exceptions import ElementPathKeyError, ElementPathTypeError
from ..helpers import collapse_white_spaces, node_position
from ..datatypes import AbstractDateTime, Duration, DayTimeDuration, \
from ..xpath_context import XPathSchemaContext
from ..namespaces import XMLNS_NAMESPACE, XSD_NAMESPACE
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode
from ..xpath_tokens import XPathToken
from .xpath1_parser import XPath1Parser
class _PrefixedReferenceToken(XPathToken):
    symbol = lookup_name = ':'
    lbp = 95
    rbp = 95

    def __init__(self, parser, value=None):
        super().__init__(parser, value)
        if self.is_spaced():
            self.lbp = self.rbp = 0
        elif self.parser.token.symbol not in ('*', '(name)', 'array'):
            self.lbp = self.rbp = 0

    def __str__(self):
        if len(self) < 2:
            return 'unparsed prefixed reference'
        elif self[1].label.endswith('function'):
            return f'{self.value!r} {self[1].label}'
        elif '*' in self.value:
            return f'{self.value!r} prefixed wildcard'
        else:
            return f'{self.value!r} prefixed name'

    @property
    def source(self) -> str:
        if self.occurrence:
            return ':'.join((tk.source for tk in self)) + self.occurrence
        else:
            return ':'.join((tk.source for tk in self))

    def led(self, left):
        version = self.parser.version
        if self.is_spaced():
            if version <= '3.0':
                raise self.wrong_syntax("a QName cannot contains spaces before or after ':'")
            return left
        if version == '1.0':
            left.expected('(name)')
        elif version <= '3.0':
            left.expected('(name)', '*')
        elif left.symbol not in ('(name)', '*'):
            return left
        if not self.parser.next_token.label.endswith('function'):
            self.parser.expected_next('(name)', '*')
        if left.symbol == '(name)':
            try:
                namespace = self.get_namespace(left.value)
            except ElementPathKeyError:
                self.parser.advance()
                self[:] = (left, self.parser.token)
                msg = 'prefix {!r} is not declared'.format(left.value)
                raise self.error('XPST0081', msg) from None
            else:
                self.parser.next_token.bind_namespace(namespace)
        elif self.parser.next_token.symbol != '(name)':
            raise self.wrong_syntax()
        self[:] = (left, self.parser.expression(95))
        if self[1].label.endswith('function'):
            self.value = f'{self[0].value}:{self[1].symbol}'
        else:
            self.value = f'{self[0].value}:{self[1].value}'
        return self

    def evaluate(self, context=None):
        if self[1].label.endswith('function'):
            return self[1].evaluate(context)
        return [x for x in self.select(context)]

    def select(self, context=None):
        if self[1].label.endswith('function'):
            value = self[1].evaluate(context)
            if isinstance(value, list):
                yield from value
            elif value is not None:
                yield value
            return
        if self[0].value == '*':
            name = '*:%s' % self[1].value
        else:
            name = '{%s}%s' % (self.get_namespace(self[0].value), self[1].value)
        if context is None:
            raise self.missing_context()
        elif isinstance(context, XPathSchemaContext):
            yield from self.select_xsd_nodes(context, name)
        elif self.xsd_types is self.parser.schema:
            for item in context.iter_children_or_self():
                if item.match_name(name):
                    yield item
        elif self.xsd_types is None or isinstance(self.xsd_types, AbstractSchemaProxy):
            for item in context.iter_children_or_self():
                if item.match_name(name):
                    assert isinstance(item, (ElementNode, AttributeNode))
                    if item.xsd_type is not None:
                        yield item
                    else:
                        xsd_node = self.parser.schema.find(item.path, self.parser.namespaces)
                        if xsd_node is not None:
                            self.add_xsd_type(xsd_node)
                        else:
                            self.xsd_types = self.parser.schema
                        context.item = self.get_typed_node(item)
                        yield context.item
        else:
            for item in context.iter_children_or_self():
                if item.match_name(name):
                    assert isinstance(item, (ElementNode, AttributeNode))
                    if item.xsd_type is not None:
                        yield item
                    else:
                        context.item = self.get_typed_node(item)
                        yield context.item