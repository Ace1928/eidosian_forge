import inspect
import logging
from os_ken import utils
from os_ken.controller import event
from os_ken.lib.packet import zebra
class EventZebraBase(event.EventBase):
    """
    The base class for Zebra protocol service event class.

    The subclasses have at least ``zclient`` and the same attributes with
    :py:class: `os_ken.lib.packet.zebra.ZebraMessage`.
    ``zclient`` is an instance of Zebra client class. See
    :py:class: `os_ken.services.protocols.zebra.client.zclient.ZClient` or
    :py:class: `os_ken.services.protocols.zebra.server.zserver.ZClient`.

    The subclasses are named as::

        ``"Event" + <Zebra message body class name>``

    For Example, if the service received ZEBRA_INTERFACE_ADD message,
    the body class should be
    :py:class: `os_ken.lib.packet.zebra.ZebraInterfaceAdd`, then the event
    class will be named as::

        "Event" + "ZebraInterfaceAdd" = "EventZebraInterfaceAdd"

    ``msg`` argument must be an instance of
    :py:class: `os_ken.lib.packet.zebra.ZebraMessage` and used to extract the
    attributes for the event classes.
    """

    def __init__(self, zclient, msg):
        super(EventZebraBase, self).__init__()
        assert isinstance(msg, zebra.ZebraMessage)
        self.__dict__ = msg.__dict__
        self.zclient = zclient

    def __repr__(self):
        m = ', '.join(['%s=%r' % (k, v) for k, v in self.__dict__.items() if not k.startswith('_')])
        return '%s(%s)' % (self.__class__.__name__, m)
    __str__ = __repr__