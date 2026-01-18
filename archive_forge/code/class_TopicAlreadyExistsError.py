import inspect
import sys
class TopicAlreadyExistsError(BrokerResponseError):
    errno = 36
    message = 'TOPIC_ALREADY_EXISTS'
    description = 'Topic with this name already exists.'