import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
@classmethod
def _authorize_token_and_login(cls, consumer_name, service_root, cache, timeout, proxy_info, authorization_engine, allow_access_levels, credential_store, credential_save_failed, version):
    """Authorize a request token. Log in with the resulting access token.

        This is the private, non-deprecated implementation of the
        deprecated method get_token_and_login(). Once
        get_token_and_login() is removed, this code can be streamlined
        and moved into its other call site, login_with().
        """
    if isinstance(consumer_name, Consumer):
        consumer = consumer_name
    else:
        consumer = SystemWideConsumer(consumer_name)
    credentials = Credentials(None)
    credentials.consumer = consumer
    if authorization_engine is None:
        authorization_engine = cls.authorization_engine_factory(service_root, consumer_name, None, allow_access_levels)
    if credential_store is None:
        credential_store = cls.credential_store_factory(credential_save_failed)
    else:
        cls._assert_login_argument_consistency('credential_save_failed', credential_save_failed, credential_store.credential_save_failed, 'credential_store')
    cached_credentials = credential_store.load(authorization_engine.unique_consumer_id)
    if cached_credentials is None:
        credentials = authorization_engine(credentials, credential_store)
    else:
        credentials = cached_credentials
        credentials.consumer.application_name = authorization_engine.application_name
    return cls(credentials, authorization_engine, credential_store, service_root, cache, timeout, proxy_info, version)