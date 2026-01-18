from boto.exception import SWFResponseError
class SWFTypeAlreadyExistsError(SWFResponseError):
    """
    Raised when when the workflow type or activity type already exists.
    """
    pass