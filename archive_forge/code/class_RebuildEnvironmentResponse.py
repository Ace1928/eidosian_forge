from datetime import datetime
from boto.compat import six
class RebuildEnvironmentResponse(Response):

    def __init__(self, response):
        response = response['RebuildEnvironmentResponse']
        super(RebuildEnvironmentResponse, self).__init__(response)