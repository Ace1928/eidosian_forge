import inspect
import sys
class StaleMetadata(KafkaError):
    retriable = True
    invalid_metadata = True