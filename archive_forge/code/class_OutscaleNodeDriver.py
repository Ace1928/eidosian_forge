import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class OutscaleNodeDriver(BaseEC2NodeDriver):
    """
    Base Outscale FCU node driver.

    Outscale per provider driver classes inherit from it.
    """
    connectionCls = OutscaleConnection
    name = 'Outscale'
    website = 'http://www.outscale.com'
    path = '/'
    signature_version = '2'
    NODE_STATE_MAP = {'pending': NodeState.PENDING, 'running': NodeState.RUNNING, 'shutting-down': NodeState.UNKNOWN, 'terminated': NodeState.TERMINATED, 'stopped': NodeState.STOPPED}

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='us-east-1', region_details=None, **kwargs):
        if hasattr(self, '_region'):
            region = getattr(self, '_region', None)
        if region_details is None:
            raise ValueError('Invalid region_details argument')
        if region not in region_details.keys():
            raise ValueError('Invalid region: %s' % region)
        self.region_name = region
        self.region_details = region_details
        details = self.region_details[region]
        self.api_name = details['api_name']
        self.country = details['country']
        self.connectionCls.host = details['endpoint']
        self._not_implemented_msg = 'This method is not supported in the Outscale driver'
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, **kwargs)

    def create_node(self, **kwargs):
        """
        Creates a new Outscale node. The ex_iamprofile keyword
        is not supported.

        @inherits: :class:`BaseEC2NodeDriver.create_node`

        :keyword    ex_keyname: The name of the key pair
        :type       ex_keyname: ``str``

        :keyword    ex_userdata: The user data
        :type       ex_userdata: ``str``

        :keyword    ex_security_groups: A list of names of security groups to
                                        assign to the node.
        :type       ex_security_groups:   ``list``

        :keyword    ex_metadata: The Key/Value metadata to associate
                                 with a node.
        :type       ex_metadata: ``dict``

        :keyword    ex_mincount: The minimum number of nodes to launch
        :type       ex_mincount: ``int``

        :keyword    ex_maxcount: The maximum number of nodes to launch
        :type       ex_maxcount: ``int``

        :keyword    ex_clienttoken: A unique identifier to ensure idempotency
        :type       ex_clienttoken: ``str``

        :keyword    ex_blockdevicemappings: ``list`` of ``dict`` block device
                    mappings.
        :type       ex_blockdevicemappings: ``list`` of ``dict``

        :keyword    ex_ebs_optimized: EBS-Optimized if True
        :type       ex_ebs_optimized: ``bool``
        """
        if 'ex_iamprofile' in kwargs:
            raise NotImplementedError('ex_iamprofile not implemented')
        return super().create_node(**kwargs)

    def ex_create_network(self, cidr_block, name=None):
        """
        Creates a network/VPC. Outscale does not support instance_tenancy.

        :param      cidr_block: The CIDR block assigned to the network
        :type       cidr_block: ``str``

        :param      name: An optional name for the network
        :type       name: ``str``

        :return:    Dictionary of network properties
        :rtype:     ``dict``
        """
        return super().ex_create_network(cidr_block, name=name)

    def ex_modify_instance_attribute(self, node, disable_api_termination=None, ebs_optimized=None, group_id=None, source_dest_check=None, user_data=None, instance_type=None, attributes=None):
        """
        Modifies node attributes.
        Ouscale supports the following attributes:
        'DisableApiTermination.Value', 'EbsOptimized', 'GroupId.n',
        'SourceDestCheck.Value', 'UserData.Value',
        'InstanceType.Value'

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      attributes: A dictionary with node attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
        attributes = attributes or {}
        if disable_api_termination is not None:
            attributes['DisableApiTermination.Value'] = disable_api_termination
        if ebs_optimized is not None:
            attributes['EbsOptimized'] = ebs_optimized
        if group_id is not None:
            attributes['GroupId.n'] = group_id
        if source_dest_check is not None:
            attributes['SourceDestCheck.Value'] = source_dest_check
        if user_data is not None:
            attributes['UserData.Value'] = user_data
        if instance_type is not None:
            attributes['InstanceType.Value'] = instance_type
        return super().ex_modify_instance_attribute(node, attributes)

    def ex_register_image(self, name, description=None, architecture=None, root_device_name=None, block_device_mapping=None):
        """
        Registers a Machine Image based off of an EBS-backed instance.
        Can also be used to create images from snapshots.

        Outscale does not support image_location, kernel_id and ramdisk_id.

        :param      name:  The name for the AMI being registered
        :type       name: ``str``

        :param      description: The description of the AMI (optional)
        :type       description: ``str``

        :param      architecture: The architecture of the AMI (i386/x86_64)
                                  (optional)
        :type       architecture: ``str``

        :param      root_device_name: The device name for the root device
                                      Required if registering an EBS-backed AMI
        :type       root_device_name: ``str``

        :param      block_device_mapping: A dictionary of the disk layout
                                          (optional)
        :type       block_device_mapping: ``dict``

        :rtype:     :class:`NodeImage`
        """
        return super().ex_register_image(name, description=description, architecture=architecture, root_device_name=root_device_name, block_device_mapping=block_device_mapping)

    def ex_copy_image(self, source_region, image, name=None, description=None):
        """
        Outscale does not support copying images.

        @inherits: :class:`EC2NodeDriver.ex_copy_image`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def ex_get_limits(self):
        """
        Outscale does not support getting limits.

        @inherits: :class:`EC2NodeDriver.ex_get_limits`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def ex_create_network_interface(self, subnet, name=None, description=None, private_ip_address=None):
        """
        Outscale does not support creating a network interface within a VPC.

        @inherits: :class:`EC2NodeDriver.ex_create_network_interface`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def ex_delete_network_interface(self, network_interface):
        """
        Outscale does not support deleting a network interface within a VPC.

        @inherits: :class:`EC2NodeDriver.ex_delete_network_interface`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def ex_attach_network_interface_to_node(self, network_interface, node, device_index):
        """
        Outscale does not support attaching a network interface.

        @inherits: :class:`EC2NodeDriver.ex_attach_network_interface_to_node`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def ex_detach_network_interface(self, attachment_id, force=False):
        """
        Outscale does not support detaching a network interface

        @inherits: :class:`EC2NodeDriver.ex_detach_network_interface`
        """
        raise NotImplementedError(self._not_implemented_msg)

    def list_sizes(self, location=None):
        """
        Lists available nodes sizes.

        This overrides the EC2 default method in order to use Outscale
        information or data.

        :rtype: ``list`` of :class:`NodeSize`
        """
        available_types = self.region_details[self.region_name]['instance_types']
        sizes = []
        for instance_type in available_types:
            attributes = OUTSCALE_INSTANCE_TYPES[instance_type]
            attributes = copy.deepcopy(attributes)
            price = get_size_price(driver_type='compute', driver_name='ec2_linux', size_id=instance_type, region=self.region_name)
            if price is None:
                attributes['price'] = None
            else:
                attributes['price'] = price
            attributes.update({'price': price})
            sizes.append(NodeSize(driver=self, **attributes))
        return sizes

    def ex_modify_instance_keypair(self, instance_id, key_name=None):
        """
        Modifies the keypair associated with a specified instance.
        Once the modification is done, you must restart the instance.

        :param      instance_id: The ID of the instance
        :type       instance_id: ``string``

        :param      key_name: The name of the keypair
        :type       key_name: ``string``
        """
        params = {'Action': 'ModifyInstanceKeypair'}
        params.update({'instanceId': instance_id})
        if key_name is not None:
            params.update({'keyName': key_name})
        response = self.connection.request(self.path, params=params, method='GET').object
        return findtext(element=response, xpath='return', namespace=OUTSCALE_NAMESPACE) == 'true'

    def _to_quota(self, elem):
        """
        To Quota
        """
        quota = {}
        for reference_quota_item in findall(element=elem, xpath='referenceQuotaSet/item', namespace=OUTSCALE_NAMESPACE):
            reference = findtext(element=reference_quota_item, xpath='reference', namespace=OUTSCALE_NAMESPACE)
            quota_set = []
            for quota_item in findall(element=reference_quota_item, xpath='quotaSet/item', namespace=OUTSCALE_NAMESPACE):
                ownerId = findtext(element=quota_item, xpath='ownerId', namespace=OUTSCALE_NAMESPACE)
                name = findtext(element=quota_item, xpath='name', namespace=OUTSCALE_NAMESPACE)
                displayName = findtext(element=quota_item, xpath='displayName', namespace=OUTSCALE_NAMESPACE)
                description = findtext(element=quota_item, xpath='description', namespace=OUTSCALE_NAMESPACE)
                groupName = findtext(element=quota_item, xpath='groupName', namespace=OUTSCALE_NAMESPACE)
                maxQuotaValue = findtext(element=quota_item, xpath='maxQuotaValue', namespace=OUTSCALE_NAMESPACE)
                usedQuotaValue = findtext(element=quota_item, xpath='usedQuotaValue', namespace=OUTSCALE_NAMESPACE)
                quota_set.append({'ownerId': ownerId, 'name': name, 'displayName': displayName, 'description': description, 'groupName': groupName, 'maxQuotaValue': maxQuotaValue, 'usedQuotaValue': usedQuotaValue})
            quota[reference] = quota_set
        return quota

    def ex_describe_quotas(self, dry_run=False, filters=None, max_results=None, marker=None):
        """
        Describes one or more of your quotas.

        :param      dry_run: dry_run
        :type       dry_run: ``bool``

        :param      filters: The filters so that the response returned includes
                             information for certain quotas only.
        :type       filters: ``dict``

        :param      max_results: The maximum number of items that can be
                                 returned in a single page (by default, 100)
        :type       max_results: ``int``

        :param      marker: Set quota marker
        :type       marker: ``string``

        :return:    (is_truncated, quota) tuple
        :rtype:     ``(bool, dict)``
        """
        if filters:
            raise NotImplementedError('quota filters are not implemented')
        if marker:
            raise NotImplementedError('quota marker is not implemented')
        params = {'Action': 'DescribeQuotas'}
        if dry_run:
            params.update({'DryRun': dry_run})
        if max_results:
            params.update({'MaxResults': max_results})
        response = self.connection.request(self.path, params=params, method='GET').object
        quota = self._to_quota(response)
        is_truncated = findtext(element=response, xpath='isTruncated', namespace=OUTSCALE_NAMESPACE)
        return (is_truncated, quota)

    def _to_product_type(self, elem):
        productTypeId = findtext(element=elem, xpath='productTypeId', namespace=OUTSCALE_NAMESPACE)
        description = findtext(element=elem, xpath='description', namespace=OUTSCALE_NAMESPACE)
        return {'productTypeId': productTypeId, 'description': description}

    def ex_get_product_type(self, image_id, snapshot_id=None):
        """
        Gets the product type of a specified OMI or snapshot.

        :param      image_id: The ID of the OMI
        :type       image_id: ``string``

        :param      snapshot_id: The ID of the snapshot
        :type       snapshot_id: ``string``

        :return:    A product type
        :rtype:     ``dict``
        """
        params = {'Action': 'GetProductType'}
        params.update({'ImageId': image_id})
        if snapshot_id is not None:
            params.update({'SnapshotId': snapshot_id})
        response = self.connection.request(self.path, params=params, method='GET').object
        product_type = self._to_product_type(response)
        return product_type

    def _to_product_types(self, elem):
        product_types = []
        for product_types_item in findall(element=elem, xpath='productTypeSet/item', namespace=OUTSCALE_NAMESPACE):
            productTypeId = findtext(element=product_types_item, xpath='productTypeId', namespace=OUTSCALE_NAMESPACE)
            description = findtext(element=product_types_item, xpath='description', namespace=OUTSCALE_NAMESPACE)
            product_types.append({'productTypeId': productTypeId, 'description': description})
        return product_types

    def ex_describe_product_types(self, filters=None):
        """
        Describes product types.

        :param      filters: The filters so that the list returned includes
                             information for certain quotas only.
        :type       filters: ``dict``

        :return:    A product types list
        :rtype:     ``list``
        """
        params = {'Action': 'DescribeProductTypes'}
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params, method='GET').object
        product_types = self._to_product_types(response)
        return product_types

    def _to_instance_types(self, elem):
        instance_types = []
        for instance_types_item in findall(element=elem, xpath='instanceTypeSet/item', namespace=OUTSCALE_NAMESPACE):
            name = findtext(element=instance_types_item, xpath='name', namespace=OUTSCALE_NAMESPACE)
            vcpu = findtext(element=instance_types_item, xpath='vcpu', namespace=OUTSCALE_NAMESPACE)
            memory = findtext(element=instance_types_item, xpath='memory', namespace=OUTSCALE_NAMESPACE)
            storageSize = findtext(element=instance_types_item, xpath='storageSize', namespace=OUTSCALE_NAMESPACE)
            storageCount = findtext(element=instance_types_item, xpath='storageCount', namespace=OUTSCALE_NAMESPACE)
            maxIpAddresses = findtext(element=instance_types_item, xpath='maxIpAddresses', namespace=OUTSCALE_NAMESPACE)
            ebsOptimizedAvailable = findtext(element=instance_types_item, xpath='ebsOptimizedAvailable', namespace=OUTSCALE_NAMESPACE)
            d = {'name': name, 'vcpu': vcpu, 'memory': memory, 'storageSize': storageSize, 'storageCount': storageCount, 'maxIpAddresses': maxIpAddresses, 'ebsOptimizedAvailable': ebsOptimizedAvailable}
            instance_types.append(d)
        return instance_types

    def ex_describe_instance_types(self, filters=None):
        """
        Describes instance types.

        :param      filters: The filters so that the list returned includes
                    information for instance types only
        :type       filters: ``dict``

        :return:    A instance types list
        :rtype:     ``list``
        """
        params = {'Action': 'DescribeInstanceTypes'}
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params, method='GET').object
        instance_types = self._to_instance_types(response)
        return instance_types