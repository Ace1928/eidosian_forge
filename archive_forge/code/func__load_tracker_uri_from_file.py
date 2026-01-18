import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def _load_tracker_uri_from_file(self):
    f = None
    try:
        f = open(self.tracker_file_name, 'r')
        uri = f.readline().strip()
        self._set_tracker_uri(uri)
    except IOError as e:
        if e.errno != errno.ENOENT:
            print("Couldn't read URI tracker file (%s): %s. Restarting upload from scratch." % (self.tracker_file_name, e.strerror))
    except InvalidUriError as e:
        print('Invalid tracker URI (%s) found in URI tracker file (%s). Restarting upload from scratch.' % (uri, self.tracker_file_name))
    finally:
        if f:
            f.close()