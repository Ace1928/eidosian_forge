import inspect
import sys
class TooManyInFlightRequests(KafkaError):
    retriable = True