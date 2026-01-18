import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
class LocalTests(unittest.TestCase):
    pass