from __future__ import absolute_import
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1.exceptions import TimeoutError
class PublishToPausedOrderingKeyException(Exception):
    """Publish attempted to paused ordering key. To resume publishing, call
    the resumePublish method on the publisher Client object with this
    ordering key. Ordering keys are paused if an unrecoverable error
    occurred during publish of a batch for that key.
    """

    def __init__(self, ordering_key: str):
        self.ordering_key = ordering_key
        super(PublishToPausedOrderingKeyException, self).__init__()