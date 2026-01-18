class ItemNotFoundException(SecretStorageException):
    """Raised when an item does not exist or has been deleted. Example of
    handling:

    >>> import secretstorage
    >>> connection = secretstorage.dbus_init()
    >>> item_path = '/not/existing/path'
    >>> try:
    ...     item = secretstorage.Item(connection, item_path)
    ... except secretstorage.ItemNotFoundException:
    ...     print('Item not found!')
    ...
    Item not found!
    """