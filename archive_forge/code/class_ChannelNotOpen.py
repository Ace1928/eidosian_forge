from struct import pack, unpack
class ChannelNotOpen(IrrecoverableConnectionError):
    """AMQP Channel Not Open Error."""
    code = 504