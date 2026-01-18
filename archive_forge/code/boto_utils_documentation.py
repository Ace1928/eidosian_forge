from botocore.session import Session
from .aws_auth import AWSRequestsAuth

        Example usage for talking to an AWS Elasticsearch Service:

        BotoAWSRequestsAuth(aws_host='search-service-foobar.us-east-1.es.amazonaws.com',
                            aws_region='us-east-1',
                            aws_service='es')

        The aws_access_key, aws_secret_access_key, and aws_token are discovered
        automatically from the environment, in the order described here:
        http://boto3.readthedocs.io/en/latest/guide/configuration.html#configuring-credentials
        