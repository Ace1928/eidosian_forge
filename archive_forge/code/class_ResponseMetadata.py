from datetime import datetime
from boto.compat import six
class ResponseMetadata(BaseObject):

    def __init__(self, response):
        super(ResponseMetadata, self).__init__()
        self.request_id = str(response['RequestId'])