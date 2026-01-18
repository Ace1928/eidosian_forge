from keystoneclient import base
class PolicyManager(base.CrudManager):
    """Manager class for manipulating Identity policies."""
    resource_class = Policy
    collection_key = 'policies'
    key = 'policy'

    def create(self, blob, type='application/json', **kwargs):
        """Create a policy.

        :param str blob: the policy document.
        :param str type: the MIME type of the policy blob.
        :param kwargs: any other attribute provided will be passed to the
                       server.

        :returns: the created policy returned from server.
        :rtype: :class:`keystoneclient.v3.policies.Policy`

        """
        return super(PolicyManager, self).create(blob=blob, type=type, **kwargs)

    def get(self, policy):
        """Retrieve a policy.

        :param policy: the policy to be retrieved from the server.
        :type policy: str or :class:`keystoneclient.v3.policies.Policy`

        :returns: the specified policy returned from server.
        :rtype: :class:`keystoneclient.v3.policies.Policy`

        """
        return super(PolicyManager, self).get(policy_id=base.getid(policy))

    def list(self, **kwargs):
        """List policies.

        :param kwargs: allows filter criteria to be passed where
                       supported by the server.

        :returns: a list of policies.
        :rtype: list of :class:`keystoneclient.v3.policies.Policy`.

        """
        return super(PolicyManager, self).list(**kwargs)

    def update(self, policy, blob=None, type=None, **kwargs):
        """Update a policy.

        :param policy: the policy to be updated on the server.
        :type policy: str or :class:`keystoneclient.v3.policies.Policy`
        :param str blob: the new policy document.
        :param str type: the new MIME type of the policy blob.
        :param kwargs: any other attribute provided will be passed to the
                       server.

        :returns: the updated policy returned from server.
        :rtype: :class:`keystoneclient.v3.policies.Policy`

        """
        return super(PolicyManager, self).update(policy_id=base.getid(policy), blob=blob, type=type, **kwargs)

    def delete(self, policy):
        """Delete a policy.

        :param policy: the policy to be deleted on the server.
        :type policy: str or :class:`keystoneclient.v3.policies.Policy`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
        return super(PolicyManager, self).delete(policy_id=base.getid(policy))