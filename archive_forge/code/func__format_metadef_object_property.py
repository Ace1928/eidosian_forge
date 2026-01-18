import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
def _format_metadef_object_property(name, metadef_property):
    return {'name': name, 'type': metadef_property.type or None, 'title': metadef_property.title or None, 'description': metadef_property.description or None, 'default': metadef_property.default or None, 'minimum': metadef_property.minimum or None, 'maximum': metadef_property.maximum or None, 'enum': metadef_property.enum or None, 'pattern': metadef_property.pattern or None, 'minLength': metadef_property.minLength or None, 'maxLength': metadef_property.maxLength or None, 'confidential': metadef_property.confidential or None, 'items': metadef_property.items or None, 'uniqueItems': metadef_property.uniqueItems or None, 'minItems': metadef_property.minItems or None, 'maxItems': metadef_property.maxItems or None, 'additionalItems': metadef_property.additionalItems or None}