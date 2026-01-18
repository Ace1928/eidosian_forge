from zope.interface import Interface
class IMailboxIMAP(IMailboxIMAPInfo):

    def getUIDValidity():
        """
        Return the unique validity identifier for this mailbox.

        @rtype: L{int}
        """

    def getUIDNext():
        """
        Return the likely UID for the next message added to this mailbox.

        @rtype: L{int}
        """

    def getUID(message):
        """
        Return the UID of a message in the mailbox

        @type message: L{int}
        @param message: The message sequence number

        @rtype: L{int}
        @return: The UID of the message.
        """

    def getMessageCount():
        """
        Return the number of messages in this mailbox.

        @rtype: L{int}
        """

    def getRecentCount():
        """
        Return the number of messages with the 'Recent' flag.

        @rtype: L{int}
        """

    def getUnseenCount():
        """
        Return the number of messages with the 'Unseen' flag.

        @rtype: L{int}
        """

    def isWriteable():
        """
        Get the read/write status of the mailbox.

        @rtype: L{int}
        @return: A true value if write permission is allowed, a false value
            otherwise.
        """

    def destroy():
        """
        Called before this mailbox is deleted, permanently.

        If necessary, all resources held by this mailbox should be cleaned up
        here. This function _must_ set the \\Noselect flag on this mailbox.
        """

    def requestStatus(names):
        """
        Return status information about this mailbox.

        Mailboxes which do not intend to do any special processing to generate
        the return value, C{statusRequestHelper} can be used to build the
        dictionary by calling the other interface methods which return the data
        for each name.

        @type names: Any iterable
        @param names: The status names to return information regarding. The
            possible values for each name are: MESSAGES, RECENT, UIDNEXT,
            UIDVALIDITY, UNSEEN.

        @rtype: L{dict} or L{Deferred}
        @return: A dictionary containing status information about the requested
            names is returned. If the process of looking this information up
            would be costly, a deferred whose callback will eventually be
            passed this dictionary is returned instead.
        """

    def addListener(listener):
        """
        Add a mailbox change listener

        @type listener: Any object which implements C{IMailboxIMAPListener}
        @param listener: An object to add to the set of those which will be
            notified when the contents of this mailbox change.
        """

    def removeListener(listener):
        """
        Remove a mailbox change listener

        @type listener: Any object previously added to and not removed from
            this mailbox as a listener.
        @param listener: The object to remove from the set of listeners.

        @raise ValueError: Raised when the given object is not a listener for
            this mailbox.
        """

    def addMessage(message, flags, date):
        """
        Add the given message to this mailbox.

        @type message: A file-like object
        @param message: The RFC822 formatted message

        @type flags: Any iterable of L{bytes}
        @param flags: The flags to associate with this message

        @type date: L{bytes}
        @param date: If specified, the date to associate with this message.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with the message id if
            the message is added successfully and whose errback is invoked
            otherwise.

        @raise ReadOnlyMailbox: Raised if this Mailbox is not open for
            read-write.
        """

    def expunge():
        """
        Remove all messages flagged \\Deleted.

        @rtype: L{list} or L{Deferred}
        @return: The list of message sequence numbers which were deleted, or a
            L{Deferred} whose callback will be invoked with such a list.

        @raise ReadOnlyMailbox: Raised if this Mailbox is not open for
            read-write.
        """

    def fetch(messages, uid):
        """
        Retrieve one or more messages.

        @type messages: C{MessageSet}
        @param messages: The identifiers of messages to retrieve information
            about

        @type uid: L{bool}
        @param uid: If true, the IDs specified in the query are UIDs; otherwise
            they are message sequence IDs.

        @rtype: Any iterable of two-tuples of message sequence numbers and
            implementors of C{IMessageIMAP}.
        """

    def store(messages, flags, mode, uid):
        """
        Set the flags of one or more messages.

        @type messages: A MessageSet object with the list of messages requested
        @param messages: The identifiers of the messages to set the flags of.

        @type flags: sequence of L{str}
        @param flags: The flags to set, unset, or add.

        @type mode: -1, 0, or 1
        @param mode: If mode is -1, these flags should be removed from the
            specified messages. If mode is 1, these flags should be added to
            the specified messages. If mode is 0, all existing flags should be
            cleared and these flags should be added.

        @type uid: L{bool}
        @param uid: If true, the IDs specified in the query are UIDs; otherwise
            they are message sequence IDs.

        @rtype: L{dict} or L{Deferred}
        @return: A L{dict} mapping message sequence numbers to sequences of
            L{str} representing the flags set on the message after this
            operation has been performed, or a L{Deferred} whose callback will
            be invoked with such a L{dict}.

        @raise ReadOnlyMailbox: Raised if this mailbox is not open for
            read-write.
        """