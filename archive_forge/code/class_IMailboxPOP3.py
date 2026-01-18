from zope.interface import Interface
class IMailboxPOP3(Interface):
    """
    An interface for mailbox access.

    Message indices are 0-based.

    @type loginDelay: L{int}
    @ivar loginDelay: The number of seconds between allowed logins for the
        user associated with this mailbox.

    @type messageExpiration: L{int}
    @ivar messageExpiration: The number of days messages in this mailbox will
        remain on the server before being deleted.
    """

    def listMessages(index=None):
        """
        Retrieve the size of a message, or, if none is specified, the size of
        each message in the mailbox.

        @type index: L{int} or L{None}
        @param index: The 0-based index of the message.

        @rtype: L{int}, sequence of L{int}, or L{Deferred <defer.Deferred>}
        @return: The number of octets in the specified message, or, if an
            index is not specified, a sequence of the number of octets for
            all messages in the mailbox or a deferred which fires with
            one of those. Any value which corresponds to a deleted message
            is set to 0.

        @raise ValueError or IndexError: When the index does not correspond to
            a message in the mailbox.  The use of ValueError is preferred.
        """

    def getMessage(index):
        """
        Retrieve a file containing the contents of a message.

        @type index: L{int}
        @param index: The 0-based index of a message.

        @rtype: file-like object
        @return: A file containing the message.

        @raise ValueError or IndexError: When the index does not correspond to
            a message in the mailbox.  The use of ValueError is preferred.
        """

    def getUidl(index):
        """
        Get a unique identifier for a message.

        @type index: L{int}
        @param index: The 0-based index of a message.

        @rtype: L{bytes}
        @return: A string of printable characters uniquely identifying the
            message for all time.

        @raise ValueError or IndexError: When the index does not correspond to
            a message in the mailbox.  The use of ValueError is preferred.
        """

    def deleteMessage(index):
        """
        Mark a message for deletion.

        This must not change the number of messages in this mailbox.  Further
        requests for the size of the deleted message should return 0.  Further
        requests for the message itself may raise an exception.

        @type index: L{int}
        @param index: The 0-based index of a message.

        @raise ValueError or IndexError: When the index does not correspond to
            a message in the mailbox.  The use of ValueError is preferred.
        """

    def undeleteMessages():
        """
        Undelete all messages marked for deletion.

        Any message which can be undeleted should be returned to its original
        position in the message sequence and retain its original UID.
        """

    def sync():
        """
        Discard the contents of any message marked for deletion.
        """