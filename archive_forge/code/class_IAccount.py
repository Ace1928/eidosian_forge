from zope.interface import Attribute, Interface
class IAccount(Interface):
    """
    I represent a user's account with a chat service.
    """
    client = Attribute('The L{IClient} currently connecting to this account, if any.')
    gatewayType = Attribute('A C{str} that identifies the protocol used by this account.')

    def __init__(accountName, autoLogin, username, password, host, port):
        """
        @type accountName: string
        @param accountName: A name to refer to the account by locally.
        @type autoLogin: boolean
        @type username: string
        @type password: string
        @type host: string
        @type port: integer
        """

    def isOnline():
        """
        Am I online?

        @rtype: boolean
        """

    def logOn(chatui):
        """
        Go on-line.

        @type chatui: Implementor of C{IChatUI}

        @rtype: L{Deferred} with an eventual L{IClient} result.
        """

    def logOff():
        """
        Sign off.
        """

    def getGroup(groupName):
        """
        @rtype: L{Group<IGroup>}
        """

    def getPerson(personName):
        """
        @rtype: L{Person<IPerson>}
        """