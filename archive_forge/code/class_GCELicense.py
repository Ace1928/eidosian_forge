import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCELicense(UuidMixin, LazyObject):
    """A GCE License used to track software usage in GCE nodes."""

    def __init__(self, name, project, driver):
        UuidMixin.__init__(self)
        self.id = name
        self.name = name
        self.project = project
        self.driver = driver
        self.charges_use_fee = None
        self.extra = None
        self._request()

    def _request(self):
        saved_request_path = self.driver.connection.request_path
        try:
            new_request_path = saved_request_path.replace(self.driver.project, self.project)
            self.driver.connection.request_path = new_request_path
            request = '/global/licenses/%s' % self.name
            response = self.driver.connection.request(request, method='GET').object
        except Exception:
            raise
        finally:
            self.driver.connection.request_path = saved_request_path
        self.extra = {'selfLink': response.get('selfLink'), 'kind': response.get('kind')}
        self.charges_use_fee = response['chargesUseFee']

    def destroy(self):
        raise LibcloudError('Can not destroy a License resource.')

    def __repr__(self):
        return '<GCELicense id="{}" name="{}" charges_use_fee="{}">'.format(self.id, self.name, self.charges_use_fee)