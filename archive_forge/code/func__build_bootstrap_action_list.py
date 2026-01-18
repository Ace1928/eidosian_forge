import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_bootstrap_action_list(self, bootstrap_actions):
    if not isinstance(bootstrap_actions, list):
        bootstrap_actions = [bootstrap_actions]
    params = {}
    for i, bootstrap_action in enumerate(bootstrap_actions):
        for key, value in six.iteritems(bootstrap_action):
            params['BootstrapActions.member.%s.%s' % (i + 1, key)] = value
    return params