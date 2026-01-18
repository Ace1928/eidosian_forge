from datetime import datetime
import sys
def _check_resource_type(self, resource_type, partial_object):
    """
        Ensure that attributes and methods defined for C{partial_object} match
        attributes and methods defined for C{resource_type}.

        @param resource_type: The resource type to check the attributes and
            methods against.
        @param partial_object: A dict with key/value pairs representing
            attributes and methods.
        """
    for name, value in partial_object.items():
        if isinstance(value, Callable):
            self._get_method(resource_type, name)
        else:
            self._check_attribute(resource_type, name, value)