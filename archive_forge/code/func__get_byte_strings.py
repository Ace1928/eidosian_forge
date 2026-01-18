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
def _get_byte_strings(*objects):
    """Gets a `bytes` string for each item in list of printable objects."""
    byte_objects = []
    for item in objects:
        if not isinstance(item, (six.binary_type, six.text_type)):
            item = str(item)
        if isinstance(item, six.binary_type):
            byte_objects.append(item)
        else:
            byte_objects.append(six.ensure_binary(item))
    return byte_objects