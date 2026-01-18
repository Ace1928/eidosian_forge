import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
class ConcurrentWriteFailure(RequestFailure):
    """
    A write failed due to another concurrent writer
    """
    pass