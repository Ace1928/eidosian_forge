import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
def _test_bucket_test_object(self, method, url, body, headers):
    """Object request."""
    if method != 'GET':
        raise NotImplementedError('%s is not implemented.' % method)
    if self.object_perms < google_storage.ObjectPermissions.READER:
        return self._FORBIDDEN
    else:
        return self._response_helper('get_object.json')