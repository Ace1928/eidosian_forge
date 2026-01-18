from manilaclient import api_versions
from manilaclient import base
class ShareTypeAccessManager(base.ManagerWithFind):
    """Manage :class:`ShareTypeAccess` resources."""
    resource_class = ShareTypeAccess

    def _do_list(self, share_type, action_name='share_type_access'):
        if share_type.is_public:
            return None
        return self._list('/types/%(st_id)s/%(action_name)s' % {'st_id': base.getid(share_type), 'action_name': action_name}, 'share_type_access')

    @api_versions.wraps('1.0', '2.6')
    def list(self, share_type, search_opts=None):
        return self._do_list(share_type, 'os-share-type-access')

    @api_versions.wraps('2.7')
    def list(self, share_type, search_opts=None):
        return self._do_list(share_type, 'share_type_access')

    def add_project_access(self, share_type, project):
        """Add a project to the given share type access list."""
        info = {'project': project}
        self._action('addProjectAccess', share_type, info)

    def remove_project_access(self, share_type, project):
        """Remove a project from the given share type access list."""
        info = {'project': project}
        self._action('removeProjectAccess', share_type, info)

    def _action(self, action, share_type, info, **kwargs):
        """Perform a share type action."""
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/types/%s/action' % base.getid(share_type)
        return self.api.client.post(url, body=body)