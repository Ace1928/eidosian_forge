from zope.interface import Interface
class IMessageIMAPCopier(Interface):

    def copy(messageObject):
        """
        Copy the given message object into this mailbox.

        The message object will be one which was previously returned by
        L{IMailboxIMAP.fetch}.

        Implementations which wish to offer better performance than the default
        implementation should implement this interface.

        If this interface is not implemented by the mailbox,
        L{IMailboxIMAP.addMessage} will be used instead.

        @rtype: L{Deferred} or L{int}
        @return: Either the UID of the message or a Deferred which fires with
            the UID when the copy finishes.
        """