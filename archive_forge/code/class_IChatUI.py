from zope.interface import Attribute, Interface
class IChatUI(Interface):

    def registerAccountClient(client):
        """
        Notifies user that an account has been signed on to.

        @type client: L{Client<IClient>}
        """

    def unregisterAccountClient(client):
        """
        Notifies user that an account has been signed off or disconnected.

        @type client: L{Client<IClient>}
        """

    def getContactsList():
        """
        @rtype: L{ContactsList}
        """

    def getConversation(person, Class, stayHidden=0):
        """
        For the given person object, returns the conversation window
        or creates and returns a new conversation window if one does not exist.

        @type person: L{Person<IPerson>}
        @type Class: L{Conversation<IConversation>} class
        @type stayHidden: boolean

        @rtype: L{Conversation<IConversation>}
        """

    def getGroupConversation(group, Class, stayHidden=0):
        """
        For the given group object, returns the group conversation window or
        creates and returns a new group conversation window if it doesn't exist.

        @type group: L{Group<interfaces.IGroup>}
        @type Class: L{Conversation<interfaces.IConversation>} class
        @type stayHidden: boolean

        @rtype: L{GroupConversation<interfaces.IGroupConversation>}
        """

    def getPerson(name, client):
        """
        Get a Person for a client.

        Duplicates L{IAccount.getPerson}.

        @type name: string
        @type client: L{Client<IClient>}

        @rtype: L{Person<IPerson>}
        """

    def getGroup(name, client):
        """
        Get a Group for a client.

        Duplicates L{IAccount.getGroup}.

        @type name: string
        @type client: L{Client<IClient>}

        @rtype: L{Group<IGroup>}
        """

    def contactChangedNick(oldnick, newnick):
        """
        For the given person, changes the person's name to newnick, and
        tells the contact list and any conversation windows with that person
        to change as well.

        @type oldnick: string
        @type newnick: string
        """