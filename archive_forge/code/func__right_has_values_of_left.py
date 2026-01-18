from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _right_has_values_of_left(left, right):
    for k, v in left.items():
        if not (not v and (k not in right or not right[k]) or (k in right and v == right[k])):
            if isinstance(v, list) and k in right:
                left_list = v
                right_list = right[k] or []
                if len(left_list) != len(right_list):
                    return False
                for list_val in left_list:
                    if list_val not in right_list:
                        if isinstance(list_val, dict) and (not list_val.get('protocol')):
                            modified_list_val = dict(list_val)
                            modified_list_val.update(protocol='tcp')
                            if modified_list_val in right_list:
                                continue
            else:
                return False
    for k, v in right.items():
        if v and k not in left:
            if k == 'essential' and v is True:
                pass
            else:
                return False
    return True