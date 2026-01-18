from keystoneauth1.exceptions import base
class RetriableConnectionFailure(Exception):
    """A mixin class that implies you can retry the most recent request."""
    pass