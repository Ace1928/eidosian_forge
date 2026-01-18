from openstack import exceptions
from openstack import resource
from openstack import utils
class ServerGroup(resource.Resource):
    resource_key = 'server_group'
    resources_key = 'server_groups'
    base_path = '/os-server-groups'
    _query_mapping = resource.QueryParameters('all_projects')
    _max_microversion = '2.64'
    allow_create = True
    allow_fetch = True
    allow_delete = True
    allow_list = True
    name = resource.Body('name')
    policies = resource.Body('policies')
    policy = resource.Body('policy')
    member_ids = resource.Body('members')
    metadata = resource.Body('metadata')
    project_id = resource.Body('project_id')
    rules = resource.Body('rules', type=dict)
    user_id = resource.Body('user_id')

    def create(self, session, prepend_key=True, base_path=None, **params):
        """Create a remote resource based on this instance.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param prepend_key: A boolean indicating whether the resource_key
            should be prepended in a resource creation request. Default to
            True.
        :param str base_path: Base part of the URI for creating resources, if
            different from :data:`~openstack.resource.Resource.base_path`.
        :param dict params: Additional params to pass.
        :return: This :class:`Resource` instance.
        :raises: :exc:`~openstack.exceptions.MethodNotSupported` if
            :data:`Resource.allow_create` is not set to ``True``.
        """
        if not self.allow_create:
            raise exceptions.MethodNotSupported(self, 'create')
        session = self._get_session(session)
        microversion = self._get_microversion(session, action='create')
        requires_id = self.create_requires_id if self.create_requires_id is not None else self.create_method == 'PUT'
        if self.create_exclude_id_from_body:
            self._body._dirty.discard('id')
        if utils.supports_microversion(session, '2.64'):
            if self.policies:
                if not self.policy and isinstance(self.policies, list):
                    self.policy = self.policies[0]
                self._body.clean(only={'policies'})
            microversion = self._max_microversion
        else:
            if self.rules:
                msg = 'API version 2.64 is required to set rules, but it is not available.'
                raise exceptions.NotSupported(msg)
            if self.policy:
                if not self.policies:
                    self.policies = [self.policy]
                self._body.clean(only={'policy'})
        if self.create_method == 'POST':
            request = self._prepare_request(requires_id=requires_id, prepend_key=prepend_key, base_path=base_path)
            response = session.post(request.url, json=request.body, headers=request.headers, microversion=microversion, params=params)
        else:
            raise exceptions.ResourceFailure('Invalid create method: %s' % self.create_method)
        has_body = self.has_body if self.create_returns_body is None else self.create_returns_body
        self.microversion = microversion
        self._translate_response(response, has_body=has_body)
        if self.has_body and self.create_returns_body is False:
            return self.fetch(session)
        return self