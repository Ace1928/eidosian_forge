import nacl.bindings

    Takes a modular crypt encoded argon2i or argon2id stored password hash
    and checks if the user provided password will hash to the same string
    when using the stored parameters

    :param password_hash: password hash serialized in modular crypt() format
    :type password_hash: bytes
    :param password: user provided password
    :type password: bytes
    :rtype: boolean

    .. versionadded:: 1.2
    