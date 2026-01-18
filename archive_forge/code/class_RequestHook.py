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
class RequestHook(object):
    """
    This can be extended and supplied to the connection object
    to gain access to request and response object after the request completes.
    One use for this would be to implement some specific request logging.
    """

    def handle_request_data(self, request, response, error=False):
        pass