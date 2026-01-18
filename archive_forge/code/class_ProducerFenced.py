import inspect
import sys
class ProducerFenced(KafkaError):
    """Another producer with the same transactional ID went online.
    NOTE: As it seems this will be raised by Broker if transaction timeout
    occurred also.
    """

    def __init__(self, msg='There is a newer producer using the same transactional_id ortransaction timeout occurred (check that processing time is below transaction_timeout_ms)'):
        super().__init__(msg)