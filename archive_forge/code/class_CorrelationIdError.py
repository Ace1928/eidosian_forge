import inspect
import sys
class CorrelationIdError(KafkaProtocolError):
    retriable = True