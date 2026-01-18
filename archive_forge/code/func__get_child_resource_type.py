from datetime import datetime
import sys
def _get_child_resource_type(self, resource_type):
    """Get the name and resource type for the entries in a collection.

        @param resource_type: The resource type for a collection.
        @return: (name, resource_type), where 'name' is the name of the child
            resource type and 'resource_type' is the corresponding resource
            type.
        """
    xml_id = self._find_representation_id(resource_type, 'get')
    representation_definition = self._application.representation_definitions[xml_id]
    [entry_links] = find_by_attribute(representation_definition.tag, 'name', 'entry_links')
    [resource_type] = list(entry_links)
    resource_type_url = resource_type.get('resource_type')
    resource_type_name = resource_type_url.split('#')[1]
    return (resource_type_name, self._application.get_resource_type(resource_type_url))