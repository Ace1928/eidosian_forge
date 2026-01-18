from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
class MockDateTime(object):

    def __init__(self):
        self.mock_now = None

    def utcnow(self):
        return self.mock_now