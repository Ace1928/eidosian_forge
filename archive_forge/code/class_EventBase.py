class EventBase(object):
    """
    The base of all event classes.

    A OSKen application can define its own event type by creating a subclass.
    """

    def __init__(self):
        super(EventBase, self).__init__()