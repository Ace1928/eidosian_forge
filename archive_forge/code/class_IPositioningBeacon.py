from zope.interface import Attribute, Interface
class IPositioningBeacon(Interface):
    """
    A positioning beacon.
    """
    identifier = Attribute('\n        A unique identifier for this beacon. The type is dependent on the\n        implementation, but must be immutable.\n        ')