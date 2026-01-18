from struct import pack, unpack
class AMQPNotImplementedError(IrrecoverableConnectionError):
    """AMQP Not Implemented Error."""
    code = 540