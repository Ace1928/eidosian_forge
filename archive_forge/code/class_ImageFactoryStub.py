from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
class ImageFactoryStub(object):

    def new_image(self, image_id=None, name=None, visibility='private', min_disk=0, min_ram=0, protected=False, owner=None, disk_format=None, container_format=None, extra_properties=None, tags=None, **other_args):
        return ImageStub(image_id, visibility=visibility, extra_properties=extra_properties, **other_args)