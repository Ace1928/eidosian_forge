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
def _object_exists(s3_client, bucket, key):
    """Check whether object exists in the specific bucket of S3.

        :param s3_client: An object with credentials to connect to S3
        :param bucket: S3 bucket name
        :param key: The image object name
        :returns: boolean value; If the value is true, the object is exist
                  if false, it is not.
        :raises: BadStoreConfiguration if cannot connect to S3 successfully
        """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except boto_exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False
        msg = 'Failed to get object info: %s' % encodeutils.exception_to_unicode(e)
        LOG.error(msg)
        raise glance_store.BadStoreConfiguration(store_name='s3', reason=msg)
    else:
        return True