from datetime import datetime
import sys
class FakeLaunchpad(object):
    """A fake Launchpad API class for unit tests that depend on L{Launchpad}.

    @param application: A C{wadllib.application.Application} instance for a
        Launchpad WADL definition file.
    """

    def __init__(self, credentials=None, service_root=None, cache=None, timeout=None, proxy_info=None, application=None):
        if application is None:
            from launchpadlib.testing.resources import get_application
            application = get_application()
        root_resource = FakeRoot(application)
        self.__dict__.update({'credentials': credentials, '_application': application, '_service_root': root_resource})

    def __setattr__(self, name, values):
        """Set sample data.

        @param name: The name of the attribute.
        @param values: A dict representing an object matching a resource
            defined in Launchpad's WADL definition.
        """
        service_root = self._service_root
        setattr(service_root, name, values)

    def __getattr__(self, name):
        """Get sample data.

        @param name: The name of the attribute.
        """
        return getattr(self._service_root, name)

    @classmethod
    def login(cls, consumer_name, token_string, access_secret, service_root=None, cache=None, timeout=None, proxy_info=None):
        """Convenience for setting up access credentials."""
        from launchpadlib.testing.resources import get_application
        return cls(object(), application=get_application())

    @classmethod
    def get_token_and_login(cls, consumer_name, service_root=None, cache=None, timeout=None, proxy_info=None):
        """Get credentials from Launchpad and log into the service root."""
        from launchpadlib.testing.resources import get_application
        return cls(object(), application=get_application())

    @classmethod
    def login_with(cls, consumer_name, service_root=None, launchpadlib_dir=None, timeout=None, proxy_info=None):
        """Log in to Launchpad with possibly cached credentials."""
        from launchpadlib.testing.resources import get_application
        return cls(object(), application=get_application())