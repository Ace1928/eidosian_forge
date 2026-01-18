from os_ken.controller.event import EventBase
class EventZClientDisconnected(EventZServerBase):
    """
    The event class for notifying the disconnection to Zebra client.
    """

    def __init__(self, zclient):
        super(EventZClientDisconnected, self).__init__()
        self.zclient = zclient