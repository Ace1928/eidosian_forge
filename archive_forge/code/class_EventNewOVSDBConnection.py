from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventNewOVSDBConnection(os_ken_event.EventBase):

    def __init__(self, client):
        super(EventNewOVSDBConnection, self).__init__()
        self.client = client

    def __str__(self):
        return '%s<system_id=%s>' % (self.__class__.__name__, self.client.system_id)

    @property
    def system_id(self):
        return self.client.system_id