from novaclient import base
class ServerExternalEventManager(base.Manager):
    resource_class = Event

    def create(self, events):
        """Create one or more server events.

        :param:events: A list of dictionaries containing 'server_uuid', 'name',
                       'status', and 'tag' (which may be absent)
        """
        body = {'events': events}
        return self._create('/os-server-external-events', body, 'events', return_raw=True)