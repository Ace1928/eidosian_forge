class LockedException(SecretStorageException):
    """Raised when an action cannot be performed because the collection
    is locked. Use :meth:`~secretstorage.collection.Collection.is_locked`
    to check if the collection is locked, and
    :meth:`~secretstorage.collection.Collection.unlock` to unlock it.
    """