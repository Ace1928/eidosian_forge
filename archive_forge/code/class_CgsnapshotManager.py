from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
class CgsnapshotManager(base.ManagerWithFind):
    """Manage :class:`Cgsnapshot` resources."""
    resource_class = Cgsnapshot

    def create(self, consistencygroup_id, name=None, description=None, user_id=None, project_id=None):
        """Creates a cgsnapshot.

        :param consistencygroup: Name or uuid of a consistency group
        :param name: Name of the cgsnapshot
        :param description: Description of the cgsnapshot
        :param user_id: User id derived from context
        :param project_id: Project id derived from context
        :rtype: :class:`Cgsnapshot`
       """
        body = {'cgsnapshot': {'consistencygroup_id': consistencygroup_id, 'name': name, 'description': description, 'user_id': user_id, 'project_id': project_id, 'status': 'creating'}}
        return self._create('/cgsnapshots', body, 'cgsnapshot')

    def get(self, cgsnapshot_id):
        """Get a cgsnapshot.

        :param cgsnapshot_id: The ID of the cgsnapshot to get.
        :rtype: :class:`Cgsnapshot`
        """
        return self._get('/cgsnapshots/%s' % cgsnapshot_id, 'cgsnapshot')

    def list(self, detailed=True, search_opts=None):
        """Lists all cgsnapshots.

        :rtype: list of :class:`Cgsnapshot`
        """
        query_string = utils.build_query_param(search_opts)
        detail = ''
        if detailed:
            detail = '/detail'
        return self._list('/cgsnapshots%s%s' % (detail, query_string), 'cgsnapshots')

    def delete(self, cgsnapshot):
        """Delete a cgsnapshot.

        :param cgsnapshot: The :class:`Cgsnapshot` to delete.
        """
        return self._delete('/cgsnapshots/%s' % base.getid(cgsnapshot))

    def update(self, cgsnapshot, **kwargs):
        """Update the name or description for a cgsnapshot.

        :param cgsnapshot: The :class:`Cgsnapshot` to update.
        """
        if not kwargs:
            return
        body = {'cgsnapshot': kwargs}
        return self._update('/cgsnapshots/%s' % base.getid(cgsnapshot), body)

    def _action(self, action, cgsnapshot, info=None, **kwargs):
        """Perform a cgsnapshot "action."
        """
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/cgsnapshots/%s/action' % base.getid(cgsnapshot)
        resp, body = self.api.client.post(url, body=body)
        return common_base.TupleWithMeta((resp, body), resp)