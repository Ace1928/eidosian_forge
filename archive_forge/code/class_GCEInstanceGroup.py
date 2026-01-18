import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCEInstanceGroup(UuidMixin):
    """GCEInstanceGroup represents the InstanceGroup resource."""

    def __init__(self, id, name, zone, driver, extra=None, network=None, subnetwork=None, named_ports=None):
        """
        :param  name:  Required. The name of the instance group. The name
                       must be 1-63 characters long, and comply with RFC1035.
        :type   name: ``str``

        :param  zone:  The URL of the zone where the instance group is
                       located.
        :type   zone: :class:`GCEZone`

        :param  network:  The URL of the network to which all instances in
                          the instance group belong.
        :type   network: :class:`GCENetwork`

        :param  subnetwork:  The URL of the subnetwork to which all instances
                             in the instance group belong.
        :type   subnetwork: :class:`GCESubnetwork`

        :param  named_ports:  Assigns a name to a port number. For example:
                              {name: "http", port: 80}  This allows the
                              system to reference ports by the assigned name
                              instead of a port number. Named ports can also
                              contain multiple ports. For example: [{name:
                              "http", port: 80},{name: "http", port: 8080}]
                              Named ports apply to all instances in this
                              instance group.
        :type   named_ports: ``"<type 'list'>"``

        """
        self.name = name
        self.zone = zone
        self.network = network
        self.subnetwork = subnetwork
        self.named_ports = named_ports
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCEInstanceGroup name="{}" zone="{}">'.format(self.name, self.zone.name)

    def destroy(self):
        """
        Destroy this InstanceGroup.

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_destroy_instancegroup(instancegroup=self)

    def add_instances(self, node_list):
        """
        Adds a list of instances to the specified instance group. All of the
        instances in the instance group must be in the same
        network/subnetwork. Read  Adding instances for more information.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where you are
                                adding instances.
        :type   instancegroup: :class:``GCEInstanceGroup``

        :param  node_list: List of nodes to add.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroup_add_instances(instancegroup=self, node_list=node_list)

    def list_instances(self):
        """
        Lists the instances in the specified instance group.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :return:  List of :class:`GCENode` objects.
        :rtype: ``list`` of :class:`GCENode` objects.
        """
        return self.driver.ex_instancegroup_list_instances(instancegroup=self)

    def remove_instances(self, node_list):
        """
        Removes one or more instances from the specified instance group,
        but does not delete those instances.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where you are
                                removng instances.
        :type   instancegroup: :class:``GCEInstanceGroup``

        :param  node_list: List of nodes to add.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroup_remove_instances(instancegroup=self, node_list=node_list)

    def set_named_ports(self, named_ports):
        """
        Sets the named ports for the specified instance group.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  named_ports:  Assigns a name to a port number. For example:
                              {name: "http", port: 80}  This allows the
                              system to reference ports by the assigned name
                              instead of a port number. Named ports can also
                              contain multiple ports. For example: [{name:
                              "http", port: 80},{name: "http", port: 8080}]
                              Named ports apply to all instances in this
                              instance group.
        :type   named_ports: ``list`` of {'name': ``str``, 'port`: ``int``}

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroup_set_named_ports(instancegroup=self, named_ports=named_ports)