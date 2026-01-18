import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
Get usage for a specific project

        :param name_or_id: project name or id
        :param start: :class:`datetime.datetime` or string. Start date in UTC
            Defaults to 2010-07-06T12:00:00Z (the date the OpenStack project
            was started)
        :param end: :class:`datetime.datetime` or string. End date in UTC.
            Defaults to now

        :returns: A :class:`~openstack.compute.v2.usage.Usage` object
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project
        