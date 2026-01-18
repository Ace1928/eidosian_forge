from datetime import datetime
import sys
class FakeRoot(FakeResource):
    """Fake root object for an application."""

    def __init__(self, application):
        """Create a L{FakeResource} for the service root of C{application}.

        @param application: A C{wadllib.application.Application} instance.
        """
        resource_type = application.get_resource_type(application.markup_url + '#service-root')
        super(FakeRoot, self).__init__(application, resource_type)