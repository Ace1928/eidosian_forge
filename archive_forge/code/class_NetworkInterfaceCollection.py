from boto.exception import BotoClientError
from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.group import Group
class NetworkInterfaceCollection(list):

    def __init__(self, *interfaces):
        self.extend(interfaces)

    def build_list_params(self, params, prefix=''):
        for i, spec in enumerate(self):
            full_prefix = '%sNetworkInterface.%s.' % (prefix, i)
            if spec.network_interface_id is not None:
                params[full_prefix + 'NetworkInterfaceId'] = str(spec.network_interface_id)
            if spec.device_index is not None:
                params[full_prefix + 'DeviceIndex'] = str(spec.device_index)
            else:
                params[full_prefix + 'DeviceIndex'] = 0
            if spec.subnet_id is not None:
                params[full_prefix + 'SubnetId'] = str(spec.subnet_id)
            if spec.description is not None:
                params[full_prefix + 'Description'] = str(spec.description)
            if spec.delete_on_termination is not None:
                params[full_prefix + 'DeleteOnTermination'] = 'true' if spec.delete_on_termination else 'false'
            if spec.secondary_private_ip_address_count is not None:
                params[full_prefix + 'SecondaryPrivateIpAddressCount'] = str(spec.secondary_private_ip_address_count)
            if spec.private_ip_address is not None:
                params[full_prefix + 'PrivateIpAddress'] = str(spec.private_ip_address)
            if spec.groups is not None:
                for j, group_id in enumerate(spec.groups):
                    query_param_key = '%sSecurityGroupId.%s' % (full_prefix, j)
                    params[query_param_key] = str(group_id)
            if spec.private_ip_addresses is not None:
                for k, ip_addr in enumerate(spec.private_ip_addresses):
                    query_param_key_prefix = '%sPrivateIpAddresses.%s' % (full_prefix, k)
                    params[query_param_key_prefix + '.PrivateIpAddress'] = str(ip_addr.private_ip_address)
                    if ip_addr.primary is not None:
                        params[query_param_key_prefix + '.Primary'] = 'true' if ip_addr.primary else 'false'
            if spec.associate_public_ip_address is not None:
                if not params[full_prefix + 'DeviceIndex'] in (0, '0'):
                    raise BotoClientError('Only the interface with device index of 0 can ' + 'be provided when using ' + "'associate_public_ip_address'.")
                if len(self) > 1:
                    raise BotoClientError('Only one interface can be provided when using ' + "'associate_public_ip_address'.")
                key = full_prefix + 'AssociatePublicIpAddress'
                if spec.associate_public_ip_address:
                    params[key] = 'true'
                else:
                    params[key] = 'false'