import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
class GoogleTestCase(LibcloudTestCase):
    """
    Assists in making Google tests hermetic and deterministic.

    Add anything that needs to be mocked here. Create a patcher with the
    suffix '_patcher'.

    e.g.
        _foo_patcher = mock.patch('module.submodule.class.foo', ...)

    Patchers are started at setUpClass and stopped at tearDownClass.

    Ideally, you should make a note in the thing being mocked, for clarity.
    """
    PATCHER_SUFFIX = '_patcher'
    _utcnow_patcher = mock.patch('libcloud.common.google._utcnow', return_value=STUB_UTCNOW)
    _authtype_is_gce_patcher = mock.patch('libcloud.common.google.GoogleAuthType._is_gce', return_value=False)
    _read_token_file_patcher = mock.patch('libcloud.common.google.GoogleOAuth2Credential._get_token_from_file', return_value=STUB_TOKEN_FROM_FILE)
    _write_token_file_patcher = mock.patch('libcloud.common.google.GoogleOAuth2Credential._write_token_to_file')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for patcher in [a for a in dir(cls) if a.endswith(cls.PATCHER_SUFFIX)]:
            getattr(cls, patcher).start()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for patcher in [a for a in dir(cls) if a.endswith(cls.PATCHER_SUFFIX)]:
            getattr(cls, patcher).stop()