import abc
from collections import abc as collections_abc
import datetime
from distutils import versionpredicate
import re
import uuid
import warnings
import copy
import iso8601
import netaddr
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import _utils
from oslo_versionedobjects import exception
class IPV4AndV6Address(IPAddress):

    @staticmethod
    def coerce(obj, attr, value):
        result = IPAddress.coerce(obj, attr, value)
        if result.version != 4 and result.version != 6:
            raise ValueError(_('Network "%(val)s" is not valid in field %(attr)s') % {'val': value, 'attr': attr})
        return result

    def get_schema(self):
        return {'oneOf': [IPV4Address().get_schema(), IPV6Address().get_schema()]}