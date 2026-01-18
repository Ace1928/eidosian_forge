from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _find_entity(self, entity_id=None, entity_class=None, match_filter=None, properties=None, entity_fetcher=None):
    """
        Finds a single matching entity that matches all the provided properties, unless an ID is specified, in which
        case it just fetches the one item
        :param entity_id: Optional ID of the entity which should be returned
        :param entity_class: Optional class of the entity which needs to be found
        :param match_filter: Optional search filter
        :param properties: Optional set of properties the entities should contain
        :param entity_fetcher: The fetcher for the entity type
        :return: The first entity matching the criteria, or None if none was found
        """
    search_filter = ''
    if entity_id:
        found_entity = entity_class(id=entity_id)
        try:
            found_entity.fetch()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Failed to fetch the specified entity by ID: {0}'.format(error))
        return found_entity
    elif match_filter:
        search_filter = match_filter
    elif properties:
        for num, property_name in enumerate(properties):
            if num > 0:
                search_filter += ' and '
            search_filter += '{0:s} == "{1}"'.format(property_name, properties[property_name])
    if entity_fetcher is not None:
        try:
            return entity_fetcher.get_first(filter=search_filter)
        except BambouHTTPError:
            pass
    return None