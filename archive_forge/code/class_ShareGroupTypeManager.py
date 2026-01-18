from manilaclient import api_versions
from manilaclient import base
class ShareGroupTypeManager(base.ManagerWithFind):
    """Manage :class:`ShareGroupType` resources."""
    resource_class = ShareGroupType

    def _create_share_group_type(self, name, share_types, is_public=False, group_specs=None):
        """Create a share group type.

        :param name: Descriptive name of the share group type
        :param share_types: list of either instances of ShareType or text
           with share type UUIDs
        :param is_public: True to create a public share group type
        :param group_specs: dict containing group spec key-value pairs
        :rtype: :class:`ShareGroupType`
        """
        if not share_types:
            raise ValueError('At least one share type must be specified when creating a share group type.')
        body = {'name': name, 'is_public': is_public, 'group_specs': group_specs or {}, 'share_types': [base.getid(share_type) for share_type in share_types]}
        return self._create(RESOURCES_PATH, {RESOURCE_NAME: body}, RESOURCE_NAME)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def create(self, name, share_types, is_public=False, group_specs=None):
        return self._create_share_group_type(name, share_types, is_public, group_specs)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def create(self, name, share_types, is_public=False, group_specs=None):
        return self._create_share_group_type(name, share_types, is_public, group_specs)

    def _get_share_group_type(self, share_group_type='default'):
        """Get a specific share group type.

        :param share_group_type: either instance of ShareGroupType, or text
           with UUID, or 'default'
        :rtype: :class:`ShareGroupType`
        """
        share_group_type_id = base.getid(share_group_type)
        url = RESOURCE_PATH % share_group_type_id
        return self._get(url, RESOURCE_NAME)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def get(self, share_group_type='default'):
        return self._get_share_group_type(share_group_type)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def get(self, share_group_type='default'):
        return self._get_share_group_type(share_group_type)

    def _list_share_group_types(self, show_all=True, search_opts=None):
        """Get a list of all share group types.

        :rtype: list of :class:`ShareGroupType`.
        """
        search_opts = search_opts or {}
        if show_all:
            search_opts['is_public'] = 'all'
        query_string = self._build_query_string(search_opts)
        url = RESOURCES_PATH + query_string
        return self._list(url, RESOURCES_NAME)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def list(self, show_all=True, search_opts=None):
        return self._list_share_group_types(show_all, search_opts)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def list(self, show_all=True, search_opts=None):
        return self._list_share_group_types(show_all, search_opts)

    def _delete_share_group_type(self, share_group_type):
        """Delete a specific share group type.

        :param share_group_type: either instance of ShareGroupType, or text
           with UUID
        """
        share_group_type_id = base.getid(share_group_type)
        url = RESOURCE_PATH % share_group_type_id
        self._delete(url)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def delete(self, share_group_type):
        self._delete_share_group_type(share_group_type)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def delete(self, share_group_type):
        self._delete_share_group_type(share_group_type)