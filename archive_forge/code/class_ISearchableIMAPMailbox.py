from zope.interface import Interface
class ISearchableIMAPMailbox(Interface):

    def search(query, uid):
        """
        Search for messages that meet the given query criteria.

        If this interface is not implemented by the mailbox,
        L{IMailboxIMAP.fetch} and various methods of L{IMessageIMAP} will be
        used instead.

        Implementations which wish to offer better performance than the default
        implementation should implement this interface.

        @type query: L{list}
        @param query: The search criteria

        @type uid: L{bool}
        @param uid: If true, the IDs specified in the query are UIDs; otherwise
            they are message sequence IDs.

        @rtype: L{list} or L{Deferred}
        @return: A list of message sequence numbers or message UIDs which match
            the search criteria or a L{Deferred} whose callback will be invoked
            with such a list.

        @raise IllegalQueryError: Raised when query is not valid.
        """