from zope.interface import Attribute, Interface
class IPerson(Interface):

    def __init__(name, account):
        """
        Initialize me.

        @param name: My name, as the server knows me.
        @type name: string
        @param account: The account I am accessed through.
        @type account: I{Account}
        """

    def isOnline():
        """
        Am I online right now?

        @rtype: boolean
        """

    def getStatus():
        """
        What is my on-line status?

        @return: L{locals.StatusEnum}
        """

    def getIdleTime():
        """
        @rtype: string (XXX: How about a scalar?)
        """

    def sendMessage(text, metadata=None):
        """
        Send a message to this person.

        @type text: string
        @type metadata: dict
        """