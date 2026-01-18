from struct import pack, unpack
class ContentTooLarge(RecoverableChannelError):
    """AMQP Content Too Large Error."""
    code = 311