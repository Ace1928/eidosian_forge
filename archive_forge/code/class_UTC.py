from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
class UTC(tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return 'Z'

    def dst(self, dt):
        return timedelta(0)