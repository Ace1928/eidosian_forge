import abc
from oslo_log import log
from keystone import exception
class CredentialDriverBase(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_credential(self, credential_id, credential):
        """Create a new credential.

        :raises keystone.exception.Conflict: If a duplicate credential exists.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_credentials(self, hints):
        """List all credentials.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.

        :returns: a list of credential_refs or an empty list.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_credentials_for_user(self, user_id, type=None):
        """List credentials for a user.

        :param user_id: ID of a user to filter credentials by.
        :param type: type of credentials to filter on.

        :returns: a list of credential_refs or an empty list.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_credential(self, credential_id):
        """Get a credential by ID.

        :returns: credential_ref
        :raises keystone.exception.CredentialNotFound: If credential doesn't
            exist.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def update_credential(self, credential_id, credential):
        """Update an existing credential.

        :raises keystone.exception.CredentialNotFound: If credential doesn't
            exist.
        :raises keystone.exception.Conflict: If a duplicate credential exists.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_credential(self, credential_id):
        """Delete an existing credential.

        :raises keystone.exception.CredentialNotFound: If credential doesn't
            exist.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_credentials_for_project(self, project_id):
        """Delete all credentials for a project."""
        self._delete_credentials(lambda cr: cr['project_id'] == project_id)

    @abc.abstractmethod
    def delete_credentials_for_user(self, user_id):
        """Delete all credentials for a user."""
        self._delete_credentials(lambda cr: cr['user_id'] == user_id)

    def _delete_credentials(self, match_fn):
        """Do the actual credential deletion work (default implementation).

        :param match_fn: function that takes a credential dict as the
                         parameter and returns true or false if the
                         identifier matches the credential dict.
        """
        for cr in self.list_credentials():
            if match_fn(cr):
                try:
                    self.credential_api.delete_credential(cr['id'])
                except exception.CredentialNotFound:
                    LOG.debug('Deletion of credential is not required: %s', cr['id'])