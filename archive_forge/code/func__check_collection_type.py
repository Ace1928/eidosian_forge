from datetime import datetime
import sys
def _check_collection_type(self, resource_type, partial_object):
    """
        Ensure that attributes and methods defined for C{partial_object} match
        attributes and methods defined for C{resource_type}.  Collection
        entries are treated specially.

        @param resource_type: The resource type to check the attributes and
            methods against.
        @param partial_object: A dict with key/value pairs representing
            attributes and methods.
        @return: (name, resource_type), where 'name' is the name of the child
            resource type and 'resource_type' is the corresponding resource
            type.
        """
    name = None
    child_resource_type = None
    for name, value in partial_object.items():
        if name == 'entries':
            name, child_resource_type = self._check_entries(resource_type, value)
        elif isinstance(value, Callable):
            self._get_method(resource_type, name)
        else:
            self._check_attribute(resource_type, name, value)
    return (name, child_resource_type)