from zope.interface import Attribute, Interface
class IInitializer(Interface):
    """
    Interface for XML stream initializers.

    Initializers perform a step in getting the XML stream ready to be
    used for the exchange of XML stanzas.
    """