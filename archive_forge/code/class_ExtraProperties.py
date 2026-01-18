from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
class ExtraProperties(abc.MutableMapping, dict):

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __eq__(self, other):
        if isinstance(other, ExtraProperties):
            return dict.__eq__(self, dict(other))
        elif isinstance(other, dict):
            return dict.__eq__(self, other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return dict.__len__(self)

    def keys(self):
        return dict.keys(self)