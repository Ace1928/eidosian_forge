from manilaclient import api_versions
from manilaclient import base
class ShareGroupTypeAccessManager(base.ManagerWithFind):
    """Manage :class:`ShareGroupTypeAccess` resources."""
    resource_class = ShareGroupTypeAccess

    def _list_share_group_type_access(self, share_group_type, search_opts=None):
        if share_group_type.is_public:
            return None
        share_group_type_id = base.getid(share_group_type)
        url = RESOURCE_PATH % share_group_type_id
        return self._list(url, RESOURCE_NAME)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def list(self, share_group_type, search_opts=None):
        return self._list_share_group_type_access(share_group_type, search_opts)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def list(self, share_group_type, search_opts=None):
        return self._list_share_group_type_access(share_group_type, search_opts)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def add_project_access(self, share_group_type, project):
        """Add a project to the given share group type access list."""
        info = {'project': project}
        self._action('addProjectAccess', share_group_type, info)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def add_project_access(self, share_group_type, project):
        """Add a project to the given share group type access list."""
        info = {'project': project}
        self._action('addProjectAccess', share_group_type, info)

    @api_versions.wraps('2.31', '2.54')
    @api_versions.experimental_api
    def remove_project_access(self, share_group_type, project):
        """Remove a project from the given share group type access list."""
        info = {'project': project}
        self._action('removeProjectAccess', share_group_type, info)

    @api_versions.wraps(SG_GRADUATION_VERSION)
    def remove_project_access(self, share_group_type, project):
        """Remove a project from the given share group type access list."""
        info = {'project': project}
        self._action('removeProjectAccess', share_group_type, info)

    def _action(self, action, share_group_type, info, **kwargs):
        """Perform a share group type action."""
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        share_group_type_id = base.getid(share_group_type)
        url = RESOURCE_PATH_ACTION % share_group_type_id
        return self.api.client.post(url, body=body)