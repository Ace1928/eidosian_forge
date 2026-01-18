from urllib import parse
from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeTypeManager(base.ManagerWithFind):
    """Manage :class:`VolumeType` resources."""
    resource_class = VolumeType

    def list(self, search_opts=None, is_public=None):
        """Lists all volume types.

        :param search_opts: Optional search filters.
        :param is_public: Whether to only get public types.
        :return: List of :class:`VolumeType`.
        """
        if not search_opts:
            search_opts = dict()
        search_opts.pop('all_tenants', None)
        if 'is_public' not in search_opts:
            search_opts['is_public'] = is_public
        query_string = '?%s' % parse.urlencode(search_opts)
        return self._list('/types%s' % query_string, 'volume_types')

    def get(self, volume_type):
        """Get a specific volume type.

        :param volume_type: The ID of the :class:`VolumeType` to get.
        :rtype: :class:`VolumeType`
        """
        return self._get('/types/%s' % base.getid(volume_type), 'volume_type')

    def default(self):
        """Get the default volume type.

        :rtype: :class:`VolumeType`
        """
        return self._get('/types/default', 'volume_type')

    def delete(self, volume_type):
        """Deletes a specific volume_type.

        :param volume_type: The name or ID of the :class:`VolumeType` to get.
        """
        return self._delete('/types/%s' % base.getid(volume_type))

    def create(self, name, description=None, is_public=True):
        """Creates a volume type.

        :param name: Descriptive name of the volume type
        :param description: Description of the volume type
        :param is_public: Volume type visibility
        :rtype: :class:`VolumeType`
        """
        body = {'volume_type': {'name': name, 'description': description, 'os-volume-type-access:is_public': is_public}}
        return self._create('/types', body, 'volume_type')

    def update(self, volume_type, name=None, description=None, is_public=None):
        """Update the name and/or description for a volume type.

        :param volume_type: The ID of the :class:`VolumeType` to update.
        :param name: Descriptive name of the volume type.
        :param description: Description of the volume type.
        :rtype: :class:`VolumeType`
        """
        body = {'volume_type': {'name': name, 'description': description}}
        if is_public is not None:
            body['volume_type']['is_public'] = is_public
        return self._update('/types/%s' % base.getid(volume_type), body, response_key='volume_type')