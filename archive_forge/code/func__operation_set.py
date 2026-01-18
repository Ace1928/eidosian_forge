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
def _operation_set(self, loc):
    """Objects and variables frequently used when operating S3 are
        returned together.

        :param loc: `glance_store.location.Location` object, supplied
                     from glance_store.location.get_location_from_uri()
        "returns: tuple of: (1) S3 client object, (2) Bucket name,
                  (3) Image Object name
        """
    return (self._create_s3_client(loc), loc.bucket, loc.key)