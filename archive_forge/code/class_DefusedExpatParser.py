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
class DefusedExpatParser(_ExpatParser):
    """
    An expat parser hardened against XML bomb attacks.

    Forbid DTDs, external entity references
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFeature(handler.feature_external_ges, False)
        self.setFeature(handler.feature_external_pes, False)

    def start_doctype_decl(self, name, sysid, pubid, has_internal_subset):
        raise DTDForbidden(name, sysid, pubid)

    def entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name):
        raise EntitiesForbidden(name, value, base, sysid, pubid, notation_name)

    def unparsed_entity_decl(self, name, base, sysid, pubid, notation_name):
        raise EntitiesForbidden(name, None, base, sysid, pubid, notation_name)

    def external_entity_ref_handler(self, context, base, sysid, pubid):
        raise ExternalReferenceForbidden(context, base, sysid, pubid)

    def reset(self):
        _ExpatParser.reset(self)
        parser = self._parser
        parser.StartDoctypeDeclHandler = self.start_doctype_decl
        parser.EntityDeclHandler = self.entity_decl
        parser.UnparsedEntityDeclHandler = self.unparsed_entity_decl
        parser.ExternalEntityRefHandler = self.external_entity_ref_handler