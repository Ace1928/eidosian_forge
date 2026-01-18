from zope.interface import Attribute, Interface
class IClient(Interface):
    account = Attribute('The L{IAccount} I am a Client for')

    def __init__(account, chatui, logonDeferred):
        """
        @type account: L{IAccount}
        @type chatui: L{IChatUI}
        @param logonDeferred: Will be called back once I am logged on.
        @type logonDeferred: L{Deferred<twisted.internet.defer.Deferred>}
        """

    def joinGroup(groupName):
        """
        @param groupName: The name of the group to join.
        @type groupName: string
        """

    def leaveGroup(groupName):
        """
        @param groupName: The name of the group to leave.
        @type groupName: string
        """

    def getGroupConversation(name, hide=0):
        pass

    def getPerson(name):
        pass