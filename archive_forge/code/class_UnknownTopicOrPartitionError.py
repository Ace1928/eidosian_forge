import inspect
import sys
class UnknownTopicOrPartitionError(BrokerResponseError):
    errno = 3
    message = 'UNKNOWN_TOPIC_OR_PARTITION'
    description = 'This request is for a topic or partition that does not exist on this broker.'
    retriable = True
    invalid_metadata = True