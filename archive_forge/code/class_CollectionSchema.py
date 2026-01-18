import jsonschema
from oslo_utils import encodeutils
from glance.common import exception
from glance.i18n import _
class CollectionSchema(object):

    def __init__(self, name, item_schema):
        self.name = name
        self.item_schema = item_schema

    def raw(self):
        definitions = None
        if self.item_schema.definitions:
            definitions = self.item_schema.definitions
            self.item_schema.definitions = None
        raw = {'name': self.name, 'properties': {self.name: {'type': 'array', 'items': self.item_schema.raw()}, 'first': {'type': 'string'}, 'next': {'type': 'string'}, 'schema': {'type': 'string'}}, 'links': [{'rel': 'first', 'href': '{first}'}, {'rel': 'next', 'href': '{next}'}, {'rel': 'describedby', 'href': '{schema}'}]}
        if definitions:
            raw['definitions'] = definitions
            self.item_schema.definitions = definitions
        return raw

    def minimal(self):
        definitions = None
        if self.item_schema.definitions:
            definitions = self.item_schema.definitions
            self.item_schema.definitions = None
        minimal = {'name': self.name, 'properties': {self.name: {'type': 'array', 'items': self.item_schema.minimal()}, 'schema': {'type': 'string'}}, 'links': [{'rel': 'describedby', 'href': '{schema}'}]}
        if definitions:
            minimal['definitions'] = definitions
            self.item_schema.definitions = definitions
        return minimal