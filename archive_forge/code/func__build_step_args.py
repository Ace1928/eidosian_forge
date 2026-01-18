import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_step_args(self, step):
    step_params = {}
    step_params['ActionOnFailure'] = step.action_on_failure
    step_params['HadoopJarStep.Jar'] = step.jar()
    main_class = step.main_class()
    if main_class:
        step_params['HadoopJarStep.MainClass'] = main_class
    args = step.args()
    if args:
        self.build_list_params(step_params, args, 'HadoopJarStep.Args.member')
    step_params['Name'] = step.name
    return step_params