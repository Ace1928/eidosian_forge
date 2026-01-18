class SecretServiceNotAvailableException(SecretStorageException):
    """Raised by :class:`~secretstorage.item.Item` or
    :class:`~secretstorage.collection.Collection` constructors, or by
    other functions in the :mod:`secretstorage.collection` module, when
    the Secret Service API is not available."""