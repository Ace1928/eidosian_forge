from zope.interface import Interface
class IRawPacketProtocol(Interface):
    """
    An interface for low-level protocols such as IP and ARP.
    """

    def addProto(num, proto):
        """
        Add a protocol on top of this one.
        """

    def datagramReceived(data, partial, dest, source, protocol):
        """
        An IP datagram has been received. Parse and process it.
        """