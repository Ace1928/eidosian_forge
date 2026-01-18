from zope.interface import Interface
class IAccountIMAP(Interface):
    """
    Interface for Account classes

    Implementors of this interface should consider implementing
    C{INamespacePresenter}.
    """

    def addMailbox(name, mbox=None):
        """
        Add a new mailbox to this account

        @type name: L{bytes}
        @param name: The name associated with this mailbox. It may not contain
            multiple hierarchical parts.

        @type mbox: An object implementing C{IMailboxIMAP}
        @param mbox: The mailbox to associate with this name. If L{None}, a
            suitable default is created and used.

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the creation succeeds, or a deferred whose
            callback will be invoked when the creation succeeds.

        @raise MailboxException: Raised if this mailbox cannot be added for
            some reason. This may also be raised asynchronously, if a
            L{Deferred} is returned.
        """

    def create(pathspec):
        """
        Create a new mailbox from the given hierarchical name.

        @type pathspec: L{bytes}
        @param pathspec: The full hierarchical name of a new mailbox to create.
            If any of the inferior hierarchical names to this one do not exist,
            they are created as well.

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the creation succeeds, or a deferred whose
            callback will be invoked when the creation succeeds.

        @raise MailboxException: Raised if this mailbox cannot be added. This
            may also be raised asynchronously, if a L{Deferred} is returned.
        """

    def select(name, rw=True):
        """
        Acquire a mailbox, given its name.

        @type name: L{bytes}
        @param name: The mailbox to acquire

        @type rw: L{bool}
        @param rw: If a true value, request a read-write version of this
            mailbox. If a false value, request a read-only version.

        @rtype: Any object implementing C{IMailboxIMAP} or L{Deferred}
        @return: The mailbox object, or a L{Deferred} whose callback will be
            invoked with the mailbox object. None may be returned if the
            specified mailbox may not be selected for any reason.
        """

    def delete(name):
        """
        Delete the mailbox with the specified name.

        @type name: L{bytes}
        @param name: The mailbox to delete.

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the mailbox is successfully deleted, or a
            L{Deferred} whose callback will be invoked when the deletion
            completes.

        @raise MailboxException: Raised if this mailbox cannot be deleted. This
            may also be raised asynchronously, if a L{Deferred} is returned.
        """

    def rename(oldname, newname):
        """
        Rename a mailbox

        @type oldname: L{bytes}
        @param oldname: The current name of the mailbox to rename.

        @type newname: L{bytes}
        @param newname: The new name to associate with the mailbox.

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the mailbox is successfully renamed, or a
            L{Deferred} whose callback will be invoked when the rename
            operation is completed.

        @raise MailboxException: Raised if this mailbox cannot be renamed. This
            may also be raised asynchronously, if a L{Deferred} is returned.
        """

    def isSubscribed(name):
        """
        Check the subscription status of a mailbox

        @type name: L{bytes}
        @param name: The name of the mailbox to check

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the given mailbox is currently subscribed to,
            a false value otherwise. A L{Deferred} may also be returned whose
            callback will be invoked with one of these values.
        """

    def subscribe(name):
        """
        Subscribe to a mailbox

        @type name: L{bytes}
        @param name: The name of the mailbox to subscribe to

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the mailbox is subscribed to successfully, or
            a Deferred whose callback will be invoked with this value when the
            subscription is successful.

        @raise MailboxException: Raised if this mailbox cannot be subscribed
            to. This may also be raised asynchronously, if a L{Deferred} is
            returned.
        """

    def unsubscribe(name):
        """
        Unsubscribe from a mailbox

        @type name: L{bytes}
        @param name: The name of the mailbox to unsubscribe from

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the mailbox is unsubscribed from successfully,
            or a Deferred whose callback will be invoked with this value when
            the unsubscription is successful.

        @raise MailboxException: Raised if this mailbox cannot be unsubscribed
            from. This may also be raised asynchronously, if a L{Deferred} is
            returned.
        """

    def listMailboxes(ref, wildcard):
        """
        List all the mailboxes that meet a certain criteria

        @type ref: L{bytes}
        @param ref: The context in which to apply the wildcard

        @type wildcard: L{bytes}
        @param wildcard: An expression against which to match mailbox names.
            '*' matches any number of characters in a mailbox name, and '%'
            matches similarly, but will not match across hierarchical
            boundaries.

        @rtype: L{list} of L{tuple}
        @return: A list of C{(mailboxName, mailboxObject)} which meet the given
            criteria. C{mailboxObject} should implement either
            C{IMailboxIMAPInfo} or C{IMailboxIMAP}. A Deferred may also be
            returned.
        """