from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
class ChatUI:
    """
    A GUI chat client.

    @type conversations: C{dict} of L{Conversation}
    @ivar conversations: A cache of all the direct windows.

    @type groupConversations: C{dict} of L{GroupConversation}
    @ivar groupConversations: A cache of all the group windows.

    @type persons: C{dict} with keys that are a C{tuple} of (C{str},
       L{IAccount<interfaces.IAccount>} provider) and values that are
       L{IPerson<interfaces.IPerson>} provider
    @ivar persons: A cache of all the users associated with this client.

    @type groups: C{dict} with keys that are a C{tuple} of (C{str},
        L{IAccount<interfaces.IAccount>} provider) and values that are
        L{IGroup<interfaces.IGroup>} provider
    @ivar groups: A cache of all the groups associated with this client.

    @type onlineClients: C{list} of L{IClient<interfaces.IClient>} providers
    @ivar onlineClients: A list of message sources currently online.

    @type contactsList: L{ContactsList}
    @ivar contactsList: A contacts list.
    """

    def __init__(self):
        self.conversations = {}
        self.groupConversations = {}
        self.persons = {}
        self.groups = {}
        self.onlineClients = []
        self.contactsList = ContactsList(self)

    def registerAccountClient(self, client):
        """
        Notify the user that an account has been signed on to.

        @type client: L{IClient<interfaces.IClient>} provider
        @param client: The client account for the person who has just signed on.

        @rtype: L{IClient<interfaces.IClient>} provider
        @return: The client, so that it may be used in a callback chain.
        """
        self.onlineClients.append(client)
        self.contactsList.registerAccountClient(client)
        return client

    def unregisterAccountClient(self, client):
        """
        Notify the user that an account has been signed off or disconnected.

        @type client: L{IClient<interfaces.IClient>} provider
        @param client: The client account for the person who has just signed
            off.
        """
        self.onlineClients.remove(client)
        self.contactsList.unregisterAccountClient(client)

    def getContactsList(self):
        """
        Get the contacts list associated with this chat window.

        @rtype: L{ContactsList}
        @return: The contacts list associated with this chat window.
        """
        return self.contactsList

    def getConversation(self, person, Class=Conversation, stayHidden=False):
        """
        For the given person object, return the conversation window or create
        and return a new conversation window if one does not exist.

        @type person: L{IPerson<interfaces.IPerson>} provider
        @param person: The person whose conversation window we want to get.

        @type Class: L{IConversation<interfaces.IConversation>} implementor
        @param Class: The kind of conversation window we want. If the conversation
            window for this person didn't already exist, create one of this type.

        @type stayHidden: C{bool}
        @param stayHidden: Whether or not the conversation window should stay
            hidden.

        @rtype: L{IConversation<interfaces.IConversation>} provider
        @return: The conversation window.
        """
        conv = self.conversations.get(person)
        if not conv:
            conv = Class(person, self)
            self.conversations[person] = conv
        if stayHidden:
            conv.hide()
        else:
            conv.show()
        return conv

    def getGroupConversation(self, group, Class=GroupConversation, stayHidden=False):
        """
        For the given group object, return the group conversation window or
        create and return a new group conversation window if it doesn't exist.

        @type group: L{IGroup<interfaces.IGroup>} provider
        @param group: The group whose conversation window we want to get.

        @type Class: L{IConversation<interfaces.IConversation>} implementor
        @param Class: The kind of conversation window we want. If the conversation
            window for this person didn't already exist, create one of this type.

        @type stayHidden: C{bool}
        @param stayHidden: Whether or not the conversation window should stay
            hidden.

        @rtype: L{IGroupConversation<interfaces.IGroupConversation>} provider
        @return: The group conversation window.
        """
        conv = self.groupConversations.get(group)
        if not conv:
            conv = Class(group, self)
            self.groupConversations[group] = conv
        if stayHidden:
            conv.hide()
        else:
            conv.show()
        return conv

    def getPerson(self, name, client):
        """
        For the given name and account client, return an instance of a
        L{IGroup<interfaces.IPerson>} provider or create and return a new
        instance of a L{IGroup<interfaces.IPerson>} provider.

        @type name: C{str}
        @param name: The name of the person of interest.

        @type client: L{IClient<interfaces.IClient>} provider
        @param client: The client account of interest.

        @rtype: L{IPerson<interfaces.IPerson>} provider
        @return: The person with that C{name}.
        """
        account = client.account
        p = self.persons.get((name, account))
        if not p:
            p = account.getPerson(name)
            self.persons[name, account] = p
        return p

    def getGroup(self, name, client):
        """
        For the given name and account client, return an instance of a
        L{IGroup<interfaces.IGroup>} provider or create and return a new instance
        of a L{IGroup<interfaces.IGroup>} provider.

        @type name: C{str}
        @param name: The name of the group of interest.

        @type client: L{IClient<interfaces.IClient>} provider
        @param client: The client account of interest.

        @rtype: L{IGroup<interfaces.IGroup>} provider
        @return: The group with that C{name}.
        """
        account = client.account
        g = self.groups.get((name, account))
        if not g:
            g = account.getGroup(name)
            self.groups[name, account] = g
        return g

    def contactChangedNick(self, person, newnick):
        """
        For the given C{person}, change the C{person}'s C{name} to C{newnick}
        and tell the contact list and any conversation windows with that
        C{person} to change as well.

        @type person: L{IPerson<interfaces.IPerson>} provider
        @param person: The person whose nickname will get changed.

        @type newnick: C{str}
        @param newnick: The new C{name} C{person} will take.
        """
        oldnick = person.name
        if (oldnick, person.account) in self.persons:
            conv = self.conversations.get(person)
            if conv:
                conv.contactChangedNick(person, newnick)
            self.contactsList.contactChangedNick(person, newnick)
            del self.persons[oldnick, person.account]
            person.name = newnick
            self.persons[person.name, person.account] = person