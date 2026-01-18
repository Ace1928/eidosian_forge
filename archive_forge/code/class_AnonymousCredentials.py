import abc
import inspect
import six
from google.auth import credentials
class AnonymousCredentials(credentials.AnonymousCredentials, Credentials):
    """Credentials that do not provide any authentication information.

    These are useful in the case of services that support anonymous access or
    local service emulators that do not use credentials. This class inherits
    from the sync anonymous credentials file, but is kept if async credentials
    is initialized and we would like anonymous credentials.
    """