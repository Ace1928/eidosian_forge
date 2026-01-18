def NullPasswordError(handler=None):
    """raised by OS crypt() supporting hashes, which forbid NULLs in password"""
    name = _get_name(handler)
    return PasswordValueError('%s does not allow NULL bytes in password' % name)