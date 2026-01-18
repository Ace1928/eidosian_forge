from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
class GroupConversation:
    """
    A GUI window of a conversation with a group of people.

    @ivar chatui: The GUI chat client associated with this conversation.
    @type chatui: L{ChatUI}

    @ivar group: The group of people that are having this conversation.
    @type group: L{IGroup<interfaces.IGroup>} provider

    @ivar members: The names of the people in this conversation.
    @type members: C{list} of C{str}
    """

    def __init__(self, group, chatui):
        """
        @param chatui: The GUI chat client associated with this conversation.
        @type chatui: L{ChatUI}

        @param group: The group of people that are having this conversation.
        @type group: L{IGroup<interfaces.IGroup>} provider
        """
        self.chatui = chatui
        self.group = group
        self.members = []

    def show(self):
        """
        Display the GroupConversationWindow.
        """
        raise NotImplementedError('Subclasses must implement this method')

    def hide(self):
        """
        Hide the GroupConversationWindow.
        """
        raise NotImplementedError('Subclasses must implement this method')

    def sendText(self, text):
        """
        Send text to the group.

        @param text: The text to be sent.
        @type text: C{str}
        """
        self.group.sendGroupMessage(text, None)

    def showGroupMessage(self, sender, text, metadata=None):
        """
        Display to the user a message sent to this group from the given sender.

        @param sender: The person sending the message.
        @type sender: C{str}

        @param text: The sent message.
        @type text: C{str}

        @param metadata: Metadata associated with this message.
        @type metadata: C{dict}
        """
        raise NotImplementedError('Subclasses must implement this method')

    def setGroupMembers(self, members):
        """
        Set the list of members in the group.

        @param members: The names of the people that will be in this group.
        @type members: C{list} of C{str}
        """
        self.members = members

    def setTopic(self, topic, author):
        """
        Change the topic for the group conversation window and display this
        change to the user.

        @param topic: This group's topic.
        @type topic: C{str}

        @param author: The person changing the topic.
        @type author: C{str}
        """
        raise NotImplementedError('Subclasses must implement this method')

    def memberJoined(self, member):
        """
        Add the given member to the list of members in the group conversation
        and displays this to the user.

        @param member: The person joining the group conversation.
        @type member: C{str}
        """
        if member not in self.members:
            self.members.append(member)

    def memberChangedNick(self, oldnick, newnick):
        """
        Change the nickname for a member of the group conversation and displays
        this change to the user.

        @param oldnick: The old nickname.
        @type oldnick: C{str}

        @param newnick: The new nickname.
        @type newnick: C{str}
        """
        if oldnick in self.members:
            self.members.remove(oldnick)
            self.members.append(newnick)

    def memberLeft(self, member):
        """
        Delete the given member from the list of members in the group
        conversation and displays the change to the user.

        @param member: The person leaving the group conversation.
        @type member: C{str}
        """
        if member in self.members:
            self.members.remove(member)