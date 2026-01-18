from zope.interface import Interface
class ICloseableMailboxIMAP(Interface):
    """
    A supplementary interface for mailboxes which require cleanup on close.

    Implementing this interface is optional. If it is implemented, the protocol
    code will call the close method defined whenever a mailbox is closed.
    """

    def close():
        """
        Close this mailbox.

        @return: A L{Deferred} which fires when this mailbox has been closed,
            or None if the mailbox can be closed immediately.
        """