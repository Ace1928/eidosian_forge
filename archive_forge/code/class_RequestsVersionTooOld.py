class RequestsVersionTooOld(Warning):
    """Used to indicate that the Requests version is too old.

    If the version of Requests is too old to support a feature, we will issue
    this warning to the user.
    """
    pass