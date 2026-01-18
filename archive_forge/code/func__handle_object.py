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
def _handle_object(self, node):
    """Convert an <object> node to a DeserializedObject."""
    Model = self._get_model_from_node(node, 'model')
    data = {}
    if node.hasAttribute('pk'):
        data[Model._meta.pk.attname] = Model._meta.pk.to_python(node.getAttribute('pk'))
    m2m_data = {}
    deferred_fields = {}
    field_names = {f.name for f in Model._meta.get_fields()}
    for field_node in node.getElementsByTagName('field'):
        field_name = field_node.getAttribute('name')
        if not field_name:
            raise base.DeserializationError("<field> node is missing the 'name' attribute")
        if self.ignore and field_name not in field_names:
            continue
        field = Model._meta.get_field(field_name)
        if field.remote_field and isinstance(field.remote_field, models.ManyToManyRel):
            value = self._handle_m2m_field_node(field_node, field)
            if value == base.DEFER_FIELD:
                deferred_fields[field] = [[getInnerText(nat_node).strip() for nat_node in obj_node.getElementsByTagName('natural')] for obj_node in field_node.getElementsByTagName('object')]
            else:
                m2m_data[field.name] = value
        elif field.remote_field and isinstance(field.remote_field, models.ManyToOneRel):
            value = self._handle_fk_field_node(field_node, field)
            if value == base.DEFER_FIELD:
                deferred_fields[field] = [getInnerText(k).strip() for k in field_node.getElementsByTagName('natural')]
            else:
                data[field.attname] = value
        else:
            if field_node.getElementsByTagName('None'):
                value = None
            else:
                value = field.to_python(getInnerText(field_node).strip())
                if field.get_internal_type() == 'JSONField':
                    value = json.loads(value, cls=field.decoder)
            data[field.name] = value
    obj = base.build_instance(Model, data, self.db)
    return base.DeserializedObject(obj, m2m_data, deferred_fields)