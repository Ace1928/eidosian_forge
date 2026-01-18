from kombu.utils.encoding import bytes_to_str
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
def _connect_to_s3(self):
    session = boto3.Session(aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key, region_name=self.aws_region)
    if session.get_credentials() is None:
        raise ImproperlyConfigured('Missing aws s3 creds')
    return session.resource('s3', endpoint_url=self.endpoint_url)