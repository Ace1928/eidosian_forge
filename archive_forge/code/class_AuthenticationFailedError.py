import inspect
import sys
class AuthenticationFailedError(KafkaError):
    retriable = False