import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def _validate_type(self, attrib, value):
    if attrib.schema.type == attrib.schema.STRING:
        if not isinstance(value, str):
            LOG.warning('Attribute %(name)s is not of type %(att_type)s', {'name': attrib.name, 'att_type': attrib.schema.STRING})
    elif attrib.schema.type == attrib.schema.LIST:
        if not isinstance(value, collections.abc.Sequence) or isinstance(value, str):
            LOG.warning('Attribute %(name)s is not of type %(att_type)s', {'name': attrib.name, 'att_type': attrib.schema.LIST})
    elif attrib.schema.type == attrib.schema.MAP:
        if not isinstance(value, collections.abc.Mapping):
            LOG.warning('Attribute %(name)s is not of type %(att_type)s', {'name': attrib.name, 'att_type': attrib.schema.MAP})
    elif attrib.schema.type == attrib.schema.INTEGER:
        if not isinstance(value, int):
            LOG.warning('Attribute %(name)s is not of type %(att_type)s', {'name': attrib.name, 'att_type': attrib.schema.INTEGER})
    elif attrib.schema.type == attrib.schema.BOOLEAN:
        try:
            strutils.bool_from_string(value, strict=True)
        except ValueError:
            LOG.warning('Attribute %(name)s is not of type %(att_type)s', {'name': attrib.name, 'att_type': attrib.schema.BOOLEAN})