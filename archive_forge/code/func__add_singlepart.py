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
def _add_singlepart(self, s3_client, image_file, bucket, key, loc, hashing_algo, verifier):
    """Stores an image file with a single part upload to S3 backend.

        :param s3_client: An object with credentials to connect to S3
        :param image_file: The image data to write, as a file-like object
        :param bucket: S3 bucket name
        :param key: The object name to be stored (image identifier)
        :param loc: `glance_store.location.Location` object, supplied
                    from glance_store.location.get_location_from_uri()
        :param hashing_algo: A hashlib algorithm identifier (string)
        :param verifier: An object used to verify signatures for images
        :returns: tuple of: (1) URL in backing store, (2) bytes written,
                  (3) checksum, (4) multihash value, and (5) a dictionary
                  with storage system specific information
        """
    os_hash_value = utils.get_hasher(hashing_algo, False)
    checksum = utils.get_hasher('md5', False)
    image_data = b''
    image_size = 0
    for chunk in utils.chunkreadable(image_file, self.WRITE_CHUNKSIZE):
        image_data += chunk
        image_size += len(chunk)
        os_hash_value.update(chunk)
        checksum.update(chunk)
        if verifier:
            verifier.update(chunk)
    s3_client.put_object(Body=image_data, Bucket=bucket, Key=key)
    hash_hex = os_hash_value.hexdigest()
    checksum_hex = checksum.hexdigest()
    metadata = {}
    if self.backend_group:
        metadata['store'] = self.backend_group
    LOG.debug('Wrote %(size)d bytes to S3 key named %(key)s with checksum %(checksum)s', {'size': image_size, 'key': key, 'checksum': checksum_hex})
    return (loc.get_uri(), image_size, checksum_hex, hash_hex, metadata)