from datetime import datetime
from boto.compat import six
class DeleteApplicationVersionResponse(Response):

    def __init__(self, response):
        response = response['DeleteApplicationVersionResponse']
        super(DeleteApplicationVersionResponse, self).__init__(response)