from datetime import datetime
import sys
def _check_attribute(self, resource_type, name, value):
    """
        Ensure that C{value} is a valid C{name} attribute on C{resource_type}.

        Does this by finding the representation for the default, canonical GET
        method (as opposed to the many "named" GET methods that exist.)

        @param resource_type: The resource type to check the attribute
            against.
        @param name: The name of the attribute.
        @param value: The value to check.
        """
    xml_id = self._find_representation_id(resource_type, 'get')
    self._check_attribute_representation(xml_id, name, value)