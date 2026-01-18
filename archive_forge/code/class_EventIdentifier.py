import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
class EventIdentifier(HeatIdentifier):
    """An identifier for an event."""
    RESOURCE_NAME, EVENT_ID = (ResourceIdentifier.RESOURCE_NAME, 'event_id')

    def __init__(self, tenant, stack_name, stack_id, path, event_id=None):
        """Initialise a new Event identifier based on components.

        The identifier is based on the identifier components of
        the associated resource and the event ID.
        """
        if event_id is not None:
            path = '/'.join([path.rstrip('/'), 'events', event_id])
        super(EventIdentifier, self).__init__(tenant, stack_name, stack_id, path)

    def __getattr__(self, attr):
        """Return a component of the identity when accessed as an attribute."""
        if attr == self.RESOURCE_NAME:
            return getattr(self.resource(), attr)
        if attr == self.EVENT_ID:
            return self._path_components()[-1]
        return HeatIdentifier.__getattr__(self, attr)

    def resource(self):
        """Return a HeatIdentifier for the owning resource."""
        return ResourceIdentifier(self.tenant, self.stack_name, self.stack_id, '/'.join(self._path_components()[:-2]))

    def stack(self):
        """Return a HeatIdentifier for the owning stack."""
        return self.resource().stack()