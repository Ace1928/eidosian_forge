import copy
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _compare_glue_connection_params(user_params, current_params):
    """
    Compare Glue connection params. If there is a difference, return True immediately else return False

    :param user_params: the Glue connection parameters passed by the user
    :param current_params: the Glue connection parameters currently configured
    :return: True if any parameter is mismatched else False
    """
    if 'Description' not in current_params:
        current_params['Description'] = ''
    if 'MatchCriteria' not in current_params:
        current_params['MatchCriteria'] = list()
    if 'PhysicalConnectionRequirements' not in current_params:
        current_params['PhysicalConnectionRequirements'] = dict()
        current_params['PhysicalConnectionRequirements']['SecurityGroupIdList'] = []
        current_params['PhysicalConnectionRequirements']['SubnetId'] = ''
    if 'ConnectionProperties' in user_params['ConnectionInput'] and user_params['ConnectionInput']['ConnectionProperties'] != current_params['ConnectionProperties']:
        return True
    if 'ConnectionType' in user_params['ConnectionInput'] and user_params['ConnectionInput']['ConnectionType'] != current_params['ConnectionType']:
        return True
    if 'Description' in user_params['ConnectionInput'] and user_params['ConnectionInput']['Description'] != current_params['Description']:
        return True
    if 'MatchCriteria' in user_params['ConnectionInput'] and set(user_params['ConnectionInput']['MatchCriteria']) != set(current_params['MatchCriteria']):
        return True
    if 'PhysicalConnectionRequirements' in user_params['ConnectionInput']:
        if 'SecurityGroupIdList' in user_params['ConnectionInput']['PhysicalConnectionRequirements'] and set(user_params['ConnectionInput']['PhysicalConnectionRequirements']['SecurityGroupIdList']) != set(current_params['PhysicalConnectionRequirements']['SecurityGroupIdList']):
            return True
        if 'SubnetId' in user_params['ConnectionInput']['PhysicalConnectionRequirements'] and user_params['ConnectionInput']['PhysicalConnectionRequirements']['SubnetId'] != current_params['PhysicalConnectionRequirements']['SubnetId']:
            return True
        if 'AvailabilityZone' in user_params['ConnectionInput']['PhysicalConnectionRequirements'] and user_params['ConnectionInput']['PhysicalConnectionRequirements']['AvailabilityZone'] != current_params['PhysicalConnectionRequirements']['AvailabilityZone']:
            return True
    return False