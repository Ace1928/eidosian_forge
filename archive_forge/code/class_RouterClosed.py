from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
class RouterClosed(Exception):
    """Raised in tasks waiting for a reply when the router is closed

    This will also be raised if the receiver task crashes, so tasks are not
    stuck waiting for a reply that can never come. The router object will not
    be usable after this is raised.
    """
    pass