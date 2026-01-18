import errno
import os
import re
import socket
import time
import six.moves.http_client as httplib
import boto
from boto import config, storage_uri_for_key
from boto.connection import AWSAuthConnection
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.keyfile import KeyFile
from boto.gs.key import Key as GSKey
def _save_tracker_info(self, key):
    self.etag_value_for_current_download = key.etag.strip('"\'')
    if not self.tracker_file_name:
        return
    f = None
    try:
        f = open(self.tracker_file_name, 'w')
        f.write('%s\n' % self.etag_value_for_current_download)
    except IOError as e:
        raise ResumableDownloadException("Couldn't write tracker file (%s): %s.\nThis can happenif you're using an incorrectly configured download tool\n(e.g., gsutil configured to save tracker files to an unwritable directory)" % (self.tracker_file_name, e.strerror), ResumableTransferDisposition.ABORT)
    finally:
        if f:
            f.close()