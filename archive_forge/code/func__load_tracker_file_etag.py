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
def _load_tracker_file_etag(self):
    f = None
    try:
        f = open(self.tracker_file_name, 'r')
        self.etag_value_for_current_download = f.readline().rstrip('\n')
        if len(self.etag_value_for_current_download) < self.MIN_ETAG_LEN:
            print("Couldn't read etag in tracker file (%s). Restarting download from scratch." % self.tracker_file_name)
    except IOError as e:
        if e.errno != errno.ENOENT:
            print("Couldn't read URI tracker file (%s): %s. Restarting download from scratch." % (self.tracker_file_name, e.strerror))
    finally:
        if f:
            f.close()