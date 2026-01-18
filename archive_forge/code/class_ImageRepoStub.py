from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
class ImageRepoStub(object):

    def __init__(self, fixtures):
        self.fixtures = fixtures

    def get(self, image_id):
        for f in self.fixtures:
            if f.image_id == image_id:
                return f
        else:
            raise ValueError(image_id)

    def list(self, *args, **kwargs):
        return self.fixtures

    def add(self, image):
        self.fixtures.append(image)