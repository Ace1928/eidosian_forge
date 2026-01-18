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
def GetExpiry(datetime_strategy, length_in_seconds):
    token_expiry = datetime_strategy.utcnow() + datetime.timedelta(seconds=length_in_seconds)
    return token_expiry