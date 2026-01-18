import abc
import logging
from oslo_config import cfg
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _metrics as metrics
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
def _check_version_cap(self, version):
    if not utils.version_is_compatible(self.version_cap, version):
        raise RPCVersionCapError(version=version, version_cap=self.version_cap)