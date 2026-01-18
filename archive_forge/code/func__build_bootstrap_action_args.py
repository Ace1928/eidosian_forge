import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_bootstrap_action_args(self, bootstrap_action):
    bootstrap_action_params = {}
    bootstrap_action_params['ScriptBootstrapAction.Path'] = bootstrap_action.path
    try:
        bootstrap_action_params['Name'] = bootstrap_action.name
    except AttributeError:
        pass
    args = bootstrap_action.args()
    if args:
        self.build_list_params(bootstrap_action_params, args, 'ScriptBootstrapAction.Args.member')
    return bootstrap_action_params