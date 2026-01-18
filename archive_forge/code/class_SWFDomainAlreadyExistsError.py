from boto.exception import SWFResponseError
class SWFDomainAlreadyExistsError(SWFResponseError):
    """
    Raised when when the domain already exists.
    """
    pass