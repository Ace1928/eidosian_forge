import contextlib
import os
import re
import textwrap
import time
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from novaclient import exceptions
from novaclient.i18n import _
def _get_resource_string(resource):
    if hasattr(resource, 'human_id') and resource.human_id:
        if hasattr(resource, 'id') and resource.id:
            return '%s (%s)' % (resource.human_id, resource.id)
        else:
            return resource.human_id
    elif hasattr(resource, 'id') and resource.id:
        return resource.id
    else:
        return resource