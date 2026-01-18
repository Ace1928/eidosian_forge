import io
import logging
import math
import re
import urllib
import eventlet
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _
import glance_store.location
@staticmethod
def _create_bucket(s3_client, s3_host, bucket, region_name=None):
    """Create bucket into the S3.

        :param s3_client: An object with credentials to connect to S3
        :param s3_host: S3 endpoint url
        :param bucket: S3 bucket name
        :param region_name: An optional region_name. If not provided, will try
               to compute it from s3_host
        :raises: BadStoreConfiguration if cannot connect to S3 successfully
        """
    if region_name:
        region = region_name
    else:
        region = get_s3_location(s3_host)
    try:
        s3_client.create_bucket(Bucket=bucket) if region == '' else s3_client.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': region})
    except boto_exceptions.ClientError as e:
        msg = 'Failed to add bucket to S3: %s' % encodeutils.exception_to_unicode(e)
        LOG.error(msg)
        raise glance_store.BadStoreConfiguration(store_name='s3', reason=msg)