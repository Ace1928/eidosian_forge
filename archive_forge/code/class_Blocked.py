from struct import pack, unpack
class Blocked(RecoverableConnectionError):
    """AMQP Connection Blocked Predicate."""