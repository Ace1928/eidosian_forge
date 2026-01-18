from zope.interface import Interface
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