import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
class MockRBD(object):

    class ImageExists(Exception):
        pass

    class ImageHasSnapshots(Exception):
        pass

    class ImageBusy(Exception):
        pass

    class ImageNotFound(Exception):
        pass

    class InvalidArgument(Exception):
        pass

    class NoSpace(Exception):
        pass

    class Image(object):

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def create_snap(self, *args, **kwargs):
            pass

        def remove_snap(self, *args, **kwargs):
            pass

        def set_snap(self, *args, **kwargs):
            pass

        def list_children(self, *args, **kwargs):
            pass

        def protect_snap(self, *args, **kwargs):
            pass

        def unprotect_snap(self, *args, **kwargs):
            pass

        def read(self, *args, **kwargs):
            raise NotImplementedError()

        def write(self, *args, **kwargs):
            raise NotImplementedError()

        def resize(self, *args, **kwargs):
            raise NotImplementedError()

        def discard(self, offset, length):
            raise NotImplementedError()

        def close(self):
            pass

        def list_snaps(self):
            raise NotImplementedError()

        def parent_info(self):
            raise NotImplementedError()

        def size(self):
            raise NotImplementedError()

    class RBD(object):

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            return False

        def create(self, *args, **kwargs):
            pass

        def remove(self, *args, **kwargs):
            pass

        def list(self, *args, **kwargs):
            raise NotImplementedError()

        def clone(self, *args, **kwargs):
            raise NotImplementedError()

        def trash_move(self, *args, **kwargs):
            pass
    RBD_FEATURE_LAYERING = 1