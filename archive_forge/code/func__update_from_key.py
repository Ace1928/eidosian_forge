import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
def _update_from_key(self, key):
    self._update_from_values(getattr(key, 'version_id', None), getattr(key, 'generation', None), getattr(key, 'is_latest', None), getattr(key, 'md5', None))