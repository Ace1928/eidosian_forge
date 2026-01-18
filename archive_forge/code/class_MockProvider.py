import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
class MockProvider(object):

    def __init__(self, provider):
        self.provider = provider

    def get_provider_name(self):
        return self.provider