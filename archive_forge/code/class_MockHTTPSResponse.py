import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
class MockHTTPSResponse(requests.Response):
    """Mock HTTP Response

        Defines the https replies from the mocked calls to do_request()
        """

    def __init__(self, content, status_code=200):
        super(ScaleIOConnectorTestCase.MockHTTPSResponse, self).__init__()
        self._content = content
        self.encoding = 'UTF-8'
        self.status_code = status_code

    def json(self, **kwargs):
        if isinstance(self._content, str):
            return super(ScaleIOConnectorTestCase.MockHTTPSResponse, self).json(**kwargs)
        return self._content

    @property
    def text(self):
        if not isinstance(self._content, str):
            return json.dumps(self._content)
        self._content = self._content.encode('utf-8')
        return super(ScaleIOConnectorTestCase.MockHTTPSResponse, self).text