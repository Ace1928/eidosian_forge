class VersionMismatchError(Exception):
    """Used to indicate a version mismatch in the version of requests required.

    The feature in use requires a newer version of Requests to function
    appropriately but the version installed is not sufficient.
    """
    pass