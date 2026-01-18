from zope.interface import Interface
class IMailboxIMAPListener(Interface):
    """
    Interface for objects interested in mailbox events
    """

    def modeChanged(writeable):
        """
        Indicates that the write status of a mailbox has changed.

        @type writeable: L{bool}
        @param writeable: A true value if write is now allowed, false
            otherwise.
        """

    def flagsChanged(newFlags):
        """
        Indicates that the flags of one or more messages have changed.

        @type newFlags: L{dict}
        @param newFlags: A mapping of message identifiers to tuples of flags
            now set on that message.
        """

    def newMessages(exists, recent):
        """
        Indicates that the number of messages in a mailbox has changed.

        @type exists: L{int} or L{None}
        @param exists: The total number of messages now in this mailbox. If the
            total number of messages has not changed, this should be L{None}.

        @type recent: L{int}
        @param recent: The number of messages now flagged C{\\Recent}. If the
            number of recent messages has not changed, this should be L{None}.
        """