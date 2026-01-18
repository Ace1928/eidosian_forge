import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_string_list(self, field, items):
    if not isinstance(items, list):
        items = [items]
    params = {}
    for i, item in enumerate(items):
        params['%s.member.%s' % (field, i + 1)] = item
    return params