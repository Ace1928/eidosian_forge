import abc
import keystone.conf
from keystone import exception
class PolicyDriverBase(object, metaclass=abc.ABCMeta):

    def _get_list_limit(self):
        return CONF.policy.list_limit or CONF.list_limit

    @abc.abstractmethod
    def enforce(self, context, credentials, action, target):
        """Verify that a user is authorized to perform action.

        For more information on a full implementation of this see:
        `keystone.policy.backends.rules.Policy.enforce`
        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def create_policy(self, policy_id, policy):
        """Store a policy blob.

        :raises keystone.exception.Conflict: If a duplicate policy exists.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_policies(self):
        """List all policies."""
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_policy(self, policy_id):
        """Retrieve a specific policy blob.

        :raises keystone.exception.PolicyNotFound: If the policy doesn't exist.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def update_policy(self, policy_id, policy):
        """Update a policy blob.

        :raises keystone.exception.PolicyNotFound: If the policy doesn't exist.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_policy(self, policy_id):
        """Remove a policy blob.

        :raises keystone.exception.PolicyNotFound: If the policy doesn't exist.

        """
        raise exception.NotImplemented()