import inspect
import sys
class BrokerResponseError(KafkaError):
    errno = None
    message = None
    description = None

    def __str__(self):
        """Add errno to standard KafkaError str"""
        return '[Error {0}] {1}'.format(self.errno, super(BrokerResponseError, self).__str__())