import abc
from os_ken.lib import stringify
class PacketBase(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    """A base class for a protocol (ethernet, ipv4, ...) header."""
    _TYPES = {}

    @classmethod
    def get_packet_type(cls, type_):
        """Per-protocol dict-like get method.

        Provided for convenience of protocol implementers.
        Internal use only."""
        return cls._TYPES.get(type_)

    @classmethod
    def register_packet_type(cls, cls_, type_):
        """Per-protocol dict-like set method.

        Provided for convenience of protocol implementers.
        Internal use only."""
        cls._TYPES[type_] = cls_

    def __init__(self):
        super(PacketBase, self).__init__()

    def __len__(self):
        return self._MIN_LEN

    @property
    def protocol_name(self):
        return self.__class__.__name__

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        """Decode a protocol header.

        This method is used only when decoding a packet.

        Decode a protocol header at offset 0 in bytearray *buf*.
        Returns the following three objects.

        * An object to describe the decoded header.

        * A packet_base.PacketBase subclass appropriate for the rest of
          the packet.  None when the rest of the packet should be considered
          as raw payload.

        * The rest of packet.

        """
        pass

    def serialize(self, payload, prev):
        """Encode a protocol header.

        This method is used only when encoding a packet.

        Encode a protocol header.
        Returns a bytearray which contains the header.

        *payload* is the rest of the packet which will immediately follow
        this header.

        *prev* is a packet_base.PacketBase subclass for the outer protocol
        header.  *prev* is None if the current header is the outer-most.
        For example, *prev* is ipv4 or ipv6 for tcp.serialize.
        """
        pass