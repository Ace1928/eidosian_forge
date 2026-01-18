import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
def _build_instance_metadata_url(url, version, path):
    """
    Builds an EC2 metadata URL for fetching information about an instance.

    Example:

        >>> _build_instance_metadata_url('http://169.254.169.254', 'latest', 'meta-data/')
        http://169.254.169.254/latest/meta-data/

    :type url: string
    :param url: URL to metadata service, e.g. 'http://169.254.169.254'

    :type version: string
    :param version: Version of the metadata to get, e.g. 'latest'

    :type path: string
    :param path: Path of the metadata to get, e.g. 'meta-data/'. If a trailing
                 slash is required it must be passed in with the path.

    :return: The full metadata URL
    """
    return '%s/%s/%s' % (url, version, path)