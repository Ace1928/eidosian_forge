from heat.common import exception
from heat.common.i18n import _
class PropertiesGroup(object):
    """A class for specifying properties relationships.

    Properties group allows to specify relations between properties or other
    properties groups with operators AND, OR and XOR by one-key dict with list
    value. For example, if there are two properties: "subprop1", which is
    child of property "prop1", and property "prop2", and they should not be
    specified together, then properties group for them should be next::

      {XOR: [["prop1", "subprop1"], ["prop2"]]}

    where each property name should be set as list of strings. Also, if these
    properties are exclusive with properties "prop3" and "prop4", which should
    be specified both, then properties group will be defined such way::

      {XOR: [ ["prop1", "subprop1"], ["prop2"],
              {AND: [ ["prop3"], ["prop4"] ]} ]}

    where one-key dict with key "AND" is nested properties group.
    """

    def __init__(self, schema, properties=None):
        self._properties = properties
        self.validate_schema(schema)
        self.schema = schema

    def validate_schema(self, current_schema):
        msg = _('Properties group schema incorrectly specified.')
        if not isinstance(current_schema, dict):
            msg = _('%(msg)s Schema should be a mapping, found %(t)s instead.') % dict(msg=msg, t=type(current_schema))
            raise exception.InvalidSchemaError(message=msg)
        if len(current_schema.keys()) > 1:
            msg = _('%(msg)s Schema should be one-key dict.') % dict(msg=msg)
            raise exception.InvalidSchemaError(message=msg)
        current_key = next(iter(current_schema))
        if current_key not in OPERATORS:
            msg = _('%(msg)s Properties group schema key should be one of the operators: %(op)s.') % dict(msg=msg, op=', '.join(OPERATORS))
            raise exception.InvalidSchemaError(message=msg)
        if not isinstance(current_schema[current_key], list):
            msg = _("%(msg)s Schemas' values should be lists of properties names or nested schemas.") % dict(msg=msg)
            raise exception.InvalidSchemaError(message=msg)
        next_msg = _('%(msg)s List items should be properties list-type names with format "[prop, prop_child, prop_sub_child, ...]" or nested properties group schemas.') % dict(msg=msg)
        for item in current_schema[current_key]:
            if isinstance(item, dict):
                self.validate_schema(item)
            elif isinstance(item, list):
                for name in item:
                    if not isinstance(name, str):
                        raise exception.InvalidSchemaError(message=next_msg)
            else:
                raise exception.InvalidSchemaError(message=next_msg)