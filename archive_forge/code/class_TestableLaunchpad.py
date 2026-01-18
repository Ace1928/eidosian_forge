from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class TestableLaunchpad(Launchpad):
    """A base class for talking to the testing root service."""

    def __init__(self, credentials, authorization_engine=None, credential_store=None, service_root='test_dev', cache=None, timeout=None, proxy_info=None, version=Launchpad.DEFAULT_VERSION):
        """Provide test-friendly defaults.

        :param authorization_engine: Defaults to None, since a test
            environment can't use an authorization engine.
        :param credential_store: Defaults to None, since tests
            generally pass in fully-formed Credentials objects.
        :param service_root: Defaults to 'test_dev'.
        """
        super(TestableLaunchpad, self).__init__(credentials, authorization_engine, credential_store, service_root=service_root, cache=cache, timeout=timeout, proxy_info=proxy_info, version=version)