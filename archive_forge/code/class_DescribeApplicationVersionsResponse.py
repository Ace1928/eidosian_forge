from datetime import datetime
from boto.compat import six
class DescribeApplicationVersionsResponse(Response):

    def __init__(self, response):
        response = response['DescribeApplicationVersionsResponse']
        super(DescribeApplicationVersionsResponse, self).__init__(response)
        response = response['DescribeApplicationVersionsResult']
        self.application_versions = []
        if response['ApplicationVersions']:
            for member in response['ApplicationVersions']:
                application_version = ApplicationVersionDescription(member)
                self.application_versions.append(application_version)