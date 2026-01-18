import inspect
import sys
class KafkaError(RuntimeError):
    retriable = False
    invalid_metadata = False

    def __str__(self):
        if not self.args:
            return self.__class__.__name__
        return '{0}: {1}'.format(self.__class__.__name__, super(KafkaError, self).__str__())