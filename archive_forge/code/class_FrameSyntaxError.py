from struct import pack, unpack
class FrameSyntaxError(IrrecoverableConnectionError):
    """AMQP Frame Syntax Error."""
    code = 502