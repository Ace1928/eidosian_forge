from urllib import parse
from cinderclient import api_versions
from cinderclient import base
class GroupTypeManager(base.ManagerWithFind):
    """Manage :class:`GroupType` resources."""
    resource_class = GroupType

    @api_versions.wraps('3.11')
    def list(self, search_opts=None, is_public=None):
        """Lists all group types.

        :rtype: list of :class:`GroupType`.
        """
        if not search_opts:
            search_opts = dict()
        query_string = ''
        if 'is_public' not in search_opts:
            search_opts['is_public'] = is_public
        query_string = '?%s' % parse.urlencode(search_opts)
        return self._list('/group_types%s' % query_string, 'group_types')

    @api_versions.wraps('3.11')
    def get(self, group_type):
        """Get a specific group type.

        :param group_type: The ID of the :class:`GroupType` to get.
        :rtype: :class:`GroupType`
        """
        return self._get('/group_types/%s' % base.getid(group_type), 'group_type')

    @api_versions.wraps('3.11')
    def default(self):
        """Get the default group type.

        :rtype: :class:`GroupType`
        """
        return self._get('/group_types/default', 'group_type')

    @api_versions.wraps('3.11')
    def delete(self, group_type):
        """Deletes a specific group_type.

        :param group_type: The name or ID of the :class:`GroupType` to get.
        """
        return self._delete('/group_types/%s' % base.getid(group_type))

    @api_versions.wraps('3.11')
    def create(self, name, description=None, is_public=True):
        """Creates a group type.

        :param name: Descriptive name of the group type
        :param description: Description of the group type
        :param is_public: Group type visibility
        :rtype: :class:`GroupType`
        """
        body = {'group_type': {'name': name, 'description': description, 'is_public': is_public}}
        return self._create('/group_types', body, 'group_type')

    @api_versions.wraps('3.11')
    def update(self, group_type, name=None, description=None, is_public=None):
        """Update the name and/or description for a group type.

        :param group_type: The ID of the :class:`GroupType` to update.
        :param name: Descriptive name of the group type.
        :param description: Description of the group type.
        :rtype: :class:`GroupType`
        """
        body = {'group_type': {'name': name, 'description': description}}
        if is_public is not None:
            body['group_type']['is_public'] = is_public
        return self._update('/group_types/%s' % base.getid(group_type), body, response_key='group_type')