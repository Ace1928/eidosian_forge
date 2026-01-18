import json
from xml.dom import pulldom
from xml.sax import handler
from xml.sax.expatreader import ExpatParser as _ExpatParser
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.xmlutils import SimplerXMLGenerator, UnserializableContentError
def _start_relational_field(self, field):
    """Output the <field> element for relational fields."""
    self.indent(2)
    self.xml.startElement('field', {'name': field.name, 'rel': field.remote_field.__class__.__name__, 'to': str(field.remote_field.model._meta)})