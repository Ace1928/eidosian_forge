from oslo_utils import uuidutils
from manilaclient import api_versions
from manilaclient import base
class ShareInstanceManager(base.ManagerWithFind):
    """Manage :class:`ShareInstances` resources."""
    resource_class = ShareInstance

    @api_versions.wraps('2.3')
    def get(self, instance):
        """Get a share instance.

        :param instance: either share object or text with its ID.
        :rtype: :class:`ShareInstance`
        """
        share_id = base.getid(instance)
        return self._get('/share_instances/%s' % share_id, 'share_instance')

    @api_versions.wraps('2.3', '2.34')
    def list(self, search_opts=None):
        """List all share instances."""
        return self.do_list()

    @api_versions.wraps('2.35')
    def list(self, export_location=None, search_opts=None):
        """List all share instances."""
        return self.do_list(export_location)

    def do_list(self, export_location=None):
        """List all share instances."""
        path = '/share_instances'
        if export_location:
            if uuidutils.is_uuid_like(export_location):
                path += '?export_location_id=' + export_location
            else:
                path += '?export_location_path=' + export_location
        return self._list(path, 'share_instances')

    def _action(self, action, instance, info=None, **kwargs):
        """Perform a share instance 'action'.

        :param action: text with action name.
        :param instance: either share object or text with its ID.
        :param info: dict with data for specified 'action'.
        :param kwargs: dict with data to be provided for action hooks.
        """
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/share_instances/%s/action' % base.getid(instance)
        return self.api.client.post(url, body=body)

    def _do_force_delete(self, instance, action_name='force_delete'):
        """Delete a share instance forcibly - share status will be avoided.

        :param instance: either share instance object or text with its ID.
        """
        return self._action(action_name, base.getid(instance))

    @api_versions.wraps('2.3', '2.6')
    def force_delete(self, instance):
        return self._do_force_delete(instance, 'os-force_delete')

    @api_versions.wraps('2.7')
    def force_delete(self, instance):
        return self._do_force_delete(instance, 'force_delete')

    def _do_reset_state(self, instance, state, action_name):
        """Update the provided share instance with the provided state.

        :param instance: either share object or text with its ID.
        :param state: text with new state to set for share.
        """
        return self._action(action_name, instance, {'status': state})

    @api_versions.wraps('2.3', '2.6')
    def reset_state(self, instance, state):
        return self._do_reset_state(instance, state, 'os-reset_status')

    @api_versions.wraps('2.7')
    def reset_state(self, instance, state):
        return self._do_reset_state(instance, state, 'reset_status')