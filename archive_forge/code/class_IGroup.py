from zope.interface import Attribute, Interface
class IGroup(Interface):
    """
    A group which you may have a conversation with.

    Groups generally have a loosely-defined set of members, who may
    leave and join at any time.
    """
    name = Attribute('My C{str} name, as the server knows me.')
    account = Attribute('The L{Account<IAccount>} I am accessed through.')

    def __init__(name, account):
        """
        Initialize me.

        @param name: My name, as the server knows me.
        @type name: str
        @param account: The account I am accessed through.
        @type account: L{Account<IAccount>}
        """

    def setTopic(text):
        """
        Set this Groups topic on the server.

        @type text: string
        """

    def sendGroupMessage(text, metadata=None):
        """
        Send a message to this group.

        @type text: str

        @type metadata: dict
        @param metadata: Valid keys for this dictionary include:

            - C{'style'}: associated with one of:
                - C{'emote'}: indicates this is an action
        """

    def join():
        """
        Join this group.
        """

    def leave():
        """
        Depart this group.
        """