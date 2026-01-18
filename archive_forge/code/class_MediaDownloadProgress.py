from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
class MediaDownloadProgress(object):
    """Status of a resumable download."""

    def __init__(self, resumable_progress, total_size):
        """Constructor.

    Args:
      resumable_progress: int, bytes received so far.
      total_size: int, total bytes in complete download.
    """
        self.resumable_progress = resumable_progress
        self.total_size = total_size

    def progress(self):
        """Percent of download completed, as a float.

    Returns:
      the percentage complete as a float, returning 0.0 if the total size of
      the download is unknown.
    """
        if self.total_size is not None and self.total_size != 0:
            return float(self.resumable_progress) / float(self.total_size)
        else:
            return 0.0