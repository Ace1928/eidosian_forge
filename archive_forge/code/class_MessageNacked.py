from struct import pack, unpack
class MessageNacked(Exception):
    """Message was nacked by broker."""