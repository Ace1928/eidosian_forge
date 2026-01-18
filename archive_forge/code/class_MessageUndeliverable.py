class MessageUndeliverable(Exception):
    """Raised if message is not routed with mandatory flag"""

    def __init__(self, exception, exchange, routing_key, message):
        super(MessageUndeliverable, self).__init__()
        self.exception = exception
        self.exchange = exchange
        self.routing_key = routing_key
        self.message = message