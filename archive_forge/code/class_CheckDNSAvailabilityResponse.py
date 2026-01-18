from datetime import datetime
from boto.compat import six
class CheckDNSAvailabilityResponse(Response):

    def __init__(self, response):
        response = response['CheckDNSAvailabilityResponse']
        super(CheckDNSAvailabilityResponse, self).__init__(response)
        response = response['CheckDNSAvailabilityResult']
        self.fully_qualified_cname = str(response['FullyQualifiedCNAME'])
        self.available = bool(response['Available'])