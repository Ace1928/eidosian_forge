import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_instance_count_and_type_args(self, master_instance_type, slave_instance_type, num_instances):
    """
        Takes a master instance type (string), a slave instance type
        (string), and a number of instances. Returns a comparable dict
        for use in making a RunJobFlow request.
        """
    params = {'Instances.MasterInstanceType': master_instance_type, 'Instances.SlaveInstanceType': slave_instance_type, 'Instances.InstanceCount': num_instances}
    return params