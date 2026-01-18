from botocore.session import Session
from .aws_auth import AWSRequestsAuth
class BotoAWSRequestsAuth(AWSRequestsAuth):

    def __init__(self, aws_host, aws_region, aws_service):
        """
        Example usage for talking to an AWS Elasticsearch Service:

        BotoAWSRequestsAuth(aws_host='search-service-foobar.us-east-1.es.amazonaws.com',
                            aws_region='us-east-1',
                            aws_service='es')

        The aws_access_key, aws_secret_access_key, and aws_token are discovered
        automatically from the environment, in the order described here:
        http://boto3.readthedocs.io/en/latest/guide/configuration.html#configuring-credentials
        """
        super(BotoAWSRequestsAuth, self).__init__(None, None, aws_host, aws_region, aws_service)
        self._refreshable_credentials = Session().get_credentials()

    def get_aws_request_headers_handler(self, r):
        credentials = get_credentials(self._refreshable_credentials)
        return self.get_aws_request_headers(r, **credentials)