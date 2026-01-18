from datetime import datetime
from boto.compat import six
class CreateApplicationResponse(Response):

    def __init__(self, response):
        response = response['CreateApplicationResponse']
        super(CreateApplicationResponse, self).__init__(response)
        response = response['CreateApplicationResult']
        if response['Application']:
            self.application = ApplicationDescription(response['Application'])
        else:
            self.application = None