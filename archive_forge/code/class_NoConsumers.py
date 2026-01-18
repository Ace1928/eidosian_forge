from struct import pack, unpack
class NoConsumers(RecoverableChannelError):
    """AMQP No Consumers Error."""
    code = 313