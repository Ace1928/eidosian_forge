import inspect
import sys
class KafkaProtocolError(KafkaError):
    retriable = True