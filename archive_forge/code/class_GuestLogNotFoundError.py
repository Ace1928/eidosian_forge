from troveclient.apiclient.exceptions import *  # noqa
class GuestLogNotFoundError(Exception):
    """The specified guest log does not exist."""
    pass