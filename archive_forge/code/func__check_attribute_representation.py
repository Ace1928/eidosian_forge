from datetime import datetime
import sys
def _check_attribute_representation(self, xml_id, name, value):
    """
        Ensure that C{value} is a valid value for C{name} with the
        representation definition matching C{xml_id}.

        @param xml_id: The XML ID for the representation to check the
            attribute against.
        @param name: The name of the attribute.
        @param value: The value to check.
        @raises IntegrityError: Raised if C{name} is not a valid attribute
            name or if C{value}'s type is not valid for the attribute.
        """
    representation = self._application.representation_definitions[xml_id]
    params = {child.name: child for child in representation.params(representation)}
    if name + '_collection_link' in params:
        resource_type = self._get_resource_type(params[name + '_collection_link'])
        child_name, child_resource_type = self._check_collection_type(resource_type, value)
    elif name + '_link' in params:
        resource_type = self._get_resource_type(params[name + '_link'])
        self._check_resource_type(resource_type, value)
    else:
        param = params.get(name)
        if param is None:
            raise IntegrityError('%s not found' % name)
        if param.type is None:
            if not isinstance(value, basestring):
                raise IntegrityError('%s is not a str or unicode for %s' % (value, name))
        elif param.type == 'xsd:dateTime':
            if not isinstance(value, datetime):
                raise IntegrityError('%s is not a datetime for %s' % (value, name))