import inspect
import sys
class KafkaConnectionError(KafkaError):
    retriable = True
    invalid_metadata = True