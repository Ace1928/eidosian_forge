import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
class SafeXMLParser(PyElementTree.XMLParser):
    """
    An XMLParser that forbids entities processing. Drops the *html* argument
    that is deprecated since version 3.4.

    :param target: the target object called by the `feed()` method of the     parser, that defaults to `TreeBuilder`.
    :param encoding: if provided, its value overrides the encoding specified     in the XML file.
    """

    def __init__(self, target: Optional[Any]=None, encoding: Optional[str]=None) -> None:
        super(SafeXMLParser, self).__init__(target=target, encoding=encoding)
        self.parser.EntityDeclHandler = self.entity_declaration
        self.parser.UnparsedEntityDeclHandler = self.unparsed_entity_declaration
        self.parser.ExternalEntityRefHandler = self.external_entity_reference

    def entity_declaration(self, entity_name, is_parameter_entity, value, base, system_id, public_id, notation_name):
        raise PyElementTree.ParseError('Entities are forbidden (entity_name={!r})'.format(entity_name))

    def unparsed_entity_declaration(self, entity_name, base, system_id, public_id, notation_name):
        raise PyElementTree.ParseError('Unparsed entities are forbidden (entity_name={!r})'.format(entity_name))

    def external_entity_reference(self, context, base, system_id, public_id):
        raise PyElementTree.ParseError('External references are forbidden (system_id={!r}, public_id={!r})'.format(system_id, public_id))