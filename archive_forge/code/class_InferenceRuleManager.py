from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
class InferenceRuleManager(base.CrudManager):
    """Manager class for manipulating Identity inference rules."""
    resource_class = InferenceRule
    collection_key = 'role_inferences'
    key = 'role_inference'

    def _implied_role_url_tail(self, prior_role, implied_role):
        base_url = '/%(prior_role_id)s/implies/%(implied_role_id)s' % {'prior_role_id': base.getid(prior_role), 'implied_role_id': base.getid(implied_role)}
        return base_url

    def create(self, prior_role, implied_role):
        """Create an inference rule.

        An inference rule is comprised of two roles, a prior role and an
        implied role. The prior role will imply the implied role.

        Valid HTTP return codes:

            * 201: Resource is created successfully
            * 404: A role cannot be found
            * 409: The inference rule already exists

        :param prior_role: the role which implies ``implied_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`
        :param implied_role: the role which is implied by ``prior_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`

        :returns: a newly created role inference returned from server.
        :rtype: :class:`keystoneclient.v3.roles.InferenceRule`

        """
        url_tail = self._implied_role_url_tail(prior_role, implied_role)
        _resp, body = self.client.put('/roles' + url_tail)
        return self._prepare_return_value(_resp, self.resource_class(self, body['role_inference']))

    def delete(self, prior_role, implied_role):
        """Delete an inference rule.

        When deleting an inference rule, both roles are required. Note that
        neither role is deleted, only the inference relationship is dissolved.

        Valid HTTP return codes:

            * 204: Delete request is accepted
            * 404: A role cannot be found

        :param prior_role: the role which implies ``implied_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`
        :param implied_role: the role which is implied by ``prior_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
        url_tail = self._implied_role_url_tail(prior_role, implied_role)
        return self._delete('/roles' + url_tail)

    def get(self, prior_role, implied_role):
        """Retrieve an inference rule.

        Valid HTTP return codes:

            * 200: Inference rule is returned
            * 404: A role cannot be found

        :param prior_role: the role which implies ``implied_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`
        :param implied_role: the role which is implied by ``prior_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`

        :returns: the specified role inference returned from server.
        :rtype: :class:`keystoneclient.v3.roles.InferenceRule`

        """
        url_tail = self._implied_role_url_tail(prior_role, implied_role)
        _resp, body = self.client.get('/roles' + url_tail)
        return self._prepare_return_value(_resp, self.resource_class(self, body['role_inference']))

    def list(self, prior_role):
        """List all roles that a role may imply.

        Valid HTTP return codes:

            * 200: List of inference rules are returned
            * 404: A role cannot be found

        :param prior_role: the role which implies ``implied_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`

        :returns: the specified role inference returned from server.
        :rtype: :class:`keystoneclient.v3.roles.InferenceRule`

        """
        url_tail = '/%s/implies' % base.getid(prior_role)
        _resp, body = self.client.get('/roles' + url_tail)
        return self._prepare_return_value(_resp, self.resource_class(self, body['role_inference']))

    def check(self, prior_role, implied_role):
        """Check if an inference rule exists.

        Valid HTTP return codes:

            * 204: The rule inference exists
            * 404: A role cannot be found

        :param prior_role: the role which implies ``implied_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`
        :param implied_role: the role which is implied by ``prior_role``.
        :type role: str or :class:`keystoneclient.v3.roles.Role`

        :returns: response object with 204 status returned from server.
        :rtype: :class:`requests.models.Response`

        """
        url_tail = self._implied_role_url_tail(prior_role, implied_role)
        return self._head('/roles' + url_tail)

    def list_inference_roles(self):
        """List all rule inferences.

        Valid HTTP return codes:

            * 200: All inference rules are returned

        :param kwargs: attributes provided will be passed to the server.

        :returns: a list of inference rules.
        :rtype: list of :class:`keystoneclient.v3.roles.InferenceRule`

        """
        return super(InferenceRuleManager, self).list()

    def update(self, **kwargs):
        raise exceptions.MethodNotImplemented(_('Update not supported for rule inferences'))

    def find(self, **kwargs):
        raise exceptions.MethodNotImplemented(_('Find not supported for rule inferences'))

    def put(self, **kwargs):
        raise exceptions.MethodNotImplemented(_('Put not supported for rule inferences'))