from zope.interface import Attribute, Interface
class IXMPPHandlerCollection(Interface):
    """
    Collection of handlers.

    Contain several handlers and manage their connection.
    """

    def __iter__():
        """
        Get an iterator over all child handlers.
        """

    def addHandler(handler):
        """
        Add a child handler.

        @type handler: L{IXMPPHandler}
        """

    def removeHandler(handler):
        """
        Remove a child handler.

        @type handler: L{IXMPPHandler}
        """