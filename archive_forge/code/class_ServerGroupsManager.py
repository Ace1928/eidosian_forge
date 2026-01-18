from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
class ServerGroupsManager(base.ManagerWithFind):
    """
    Manage :class:`ServerGroup` resources.
    """
    resource_class = ServerGroup

    def list(self, all_projects=False, limit=None, offset=None):
        """Get a list of all server groups.

        :param all_projects: Lists server groups for all projects. (optional)
        :param limit: Maximum number of server groups to return. (optional)
                      Note the API server has a configurable default limit.
                      If no limit is specified here or limit is larger than
                      default, the default limit will be used.
        :param offset: Use with `limit` to return a slice of server
                       groups. `offset` is where to start in the groups
                       list. (optional)
        :returns: list of :class:`ServerGroup`.
        """
        qparams = {}
        if all_projects:
            qparams['all_projects'] = bool(all_projects)
        if limit:
            qparams['limit'] = int(limit)
        if offset:
            qparams['offset'] = int(offset)
        return self._list('/os-server-groups', 'server_groups', filters=qparams)

    def get(self, id):
        """Get a specific server group.

        :param id: The ID of the :class:`ServerGroup` to get.
        :rtype: :class:`ServerGroup`
        """
        return self._get('/os-server-groups/%s' % id, 'server_group')

    def delete(self, id):
        """Delete a specific server group.

        :param id: The ID of the :class:`ServerGroup` to delete.
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        return self._delete('/os-server-groups/%s' % id)

    @api_versions.wraps('2.0', '2.63')
    def create(self, name, policies):
        """Create (allocate) a server group.

        :param name: The name of the server group.
        :param policies: Policy name or a list of exactly one policy name to
            associate with the server group.
        :rtype: list of :class:`ServerGroup`
        """
        policies = policies if isinstance(policies, list) else [policies]
        body = {'server_group': {'name': name, 'policies': policies}}
        return self._create('/os-server-groups', body, 'server_group')

    @api_versions.wraps('2.64')
    def create(self, name, policy, rules=None):
        """Create (allocate) a server group.

        :param name: The name of the server group.
        :param policy: Policy name to associate with the server group.
        :param rules: The rules of policy which is a dict, can be applied to
            the policy, now only ``max_server_per_host`` for ``anti-affinity``
            policy would be supported (optional).
        :rtype: list of :class:`ServerGroup`
        """
        body = {'server_group': {'name': name, 'policy': policy}}
        if rules:
            key = 'max_server_per_host'
            try:
                if key in rules:
                    rules[key] = int(rules[key])
            except ValueError:
                msg = _("Invalid '%(key)s' value: %(value)s")
                raise exceptions.CommandError(msg % {'key': key, 'value': rules[key]})
            body['server_group']['rules'] = rules
        return self._create('/os-server-groups', body, 'server_group')