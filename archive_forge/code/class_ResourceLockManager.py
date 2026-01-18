from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
class ResourceLockManager(base.ManagerWithFind):
    """Manage :class:`ResourceLock` resources."""
    resource_class = ResourceLock

    @api_versions.wraps(constants.RESOURCE_LOCK_VERSION)
    def create(self, resource_id, resource_type, resource_action='delete', lock_reason=None):
        """Creates a resource lock.

        :param resource_id: The ID of the resource to lock
        :param resource_type: The type of the resource (e.g., "share",
        "access")
        :param resource_action: The functionality to lock (e.g., "delete",
        "view/delete")
        :param lock_reason: Lock description
        :rtype: :class:`ResourceLock`
        """
        body = {'resource_lock': {'resource_id': resource_id, 'resource_type': resource_type, 'resource_action': resource_action, 'lock_reason': lock_reason}}
        return self._create('/resource-locks', body, 'resource_lock')

    @api_versions.wraps(constants.RESOURCE_LOCK_VERSION)
    def get(self, lock_id):
        """Show details of a resource lock.

        :param lock_id: The ID of the resource lock to display.
        :rtype: :class:`ResourceLock`
        """
        return self._get('/resource-locks/%s' % lock_id, 'resource_lock')

    @api_versions.wraps(constants.RESOURCE_LOCK_VERSION)
    def list(self, search_opts=None, sort_key=None, sort_dir=None):
        """Get a list of all resource locks.

        :param search_opts: Filtering options as a dictionary.
        :param sort_key: Key to be sorted (i.e. 'created_at').
        :param sort_dir: Sort direction, should be 'desc' or 'asc'.
        :rtype: list of :class:`ResourceLock`
        """
        search_opts = search_opts or {}
        sort_key = sort_key or 'created_at'
        if sort_key in constants.RESOURCE_LOCK_SORT_KEY_VALUES:
            search_opts['sort_key'] = sort_key
        else:
            raise ValueError('sort_key must be one of the following: %s.' % ', '.join(constants.RESOURCE_LOCK_SORT_KEY_VALUES))
        sort_dir = sort_dir or 'desc'
        if sort_dir in constants.SORT_DIR_VALUES:
            search_opts['sort_dir'] = sort_dir
        else:
            raise ValueError('sort_dir must be one of the following: %s.' % ', '.join(constants.SORT_DIR_VALUES))
        query_string = self._build_query_string(search_opts)
        path = '/resource-locks%s' % (query_string,)
        return self._list(path, 'resource_locks')

    @api_versions.wraps(constants.RESOURCE_LOCK_VERSION)
    def update(self, lock, **kwargs):
        """Updates a resource lock.

        :param lock: The :class:`ResourceLock` object, or a lock id to update.
        :param kwargs: "resource_action" and "lock_reason" are allowed kwargs
        :rtype: :class:`ResourceLock`
        """
        if not kwargs:
            return
        body = {'resource_lock': {}}
        if 'lock_reason' in kwargs:
            body['resource_lock']['lock_reason'] = kwargs['lock_reason']
        if 'resource_action' in kwargs:
            body['resource_lock']['resource_action'] = kwargs['resource_action']
        lock_id = base.getid(lock)
        return self._update('/resource-locks/%s' % lock_id, body)

    @api_versions.wraps(constants.RESOURCE_LOCK_VERSION)
    def delete(self, lock):
        """Delete a resource lock.

        :param lock: The :class:`ResourceLock` object, or a lock id to delete.
        """
        return self._delete('/resource-locks/%s' % base.getid(lock))