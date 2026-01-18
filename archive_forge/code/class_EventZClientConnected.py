from os_ken.controller.event import EventBase
class EventZClientConnected(EventZServerBase):
    """
    The event class for notifying the connection from Zebra client.
    """

    def __init__(self, zclient):
        super(EventZClientConnected, self).__init__()
        self.zclient = zclient