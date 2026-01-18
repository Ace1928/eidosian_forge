from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def _hashable_policy(policy, policy_list):
    """
    Takes a policy and returns a list, the contents of which are all hashable and sorted.
    Example input policy:
    {'Version': '2012-10-17',
     'Statement': [{'Action': 's3:PutObjectAcl',
                    'Sid': 'AddCannedAcl2',
                    'Resource': 'arn:aws:s3:::test_policy/*',
                    'Effect': 'Allow',
                    'Principal': {'AWS': ['arn:aws:iam::XXXXXXXXXXXX:user/username1', 'arn:aws:iam::XXXXXXXXXXXX:user/username2']}
                   }]}
    Returned value:
    [('Statement',  ((('Action', ('s3:PutObjectAcl',)),
                      ('Effect', ('Allow',)),
                      ('Principal', ('AWS', (('arn:aws:iam::XXXXXXXXXXXX:user/username1',), ('arn:aws:iam::XXXXXXXXXXXX:user/username2',)))),
                      ('Resource', ('arn:aws:s3:::test_policy/*',)), ('Sid', ('AddCannedAcl2',)))),
     ('Version', ('2012-10-17',)))]

    """
    if isinstance(policy, bool):
        return tuple([str(policy).lower()])
    elif isinstance(policy, int):
        return tuple([str(policy)])
    if isinstance(policy, list):
        for each in policy:
            hashed_policy = _hashable_policy(each, [])
            tupleified = _tuplify_list(hashed_policy)
            policy_list.append(tupleified)
    elif isinstance(policy, string_types) or isinstance(policy, binary_type):
        policy = to_text(policy)
        policy = _canonify_root_arn(policy)
        return [policy]
    elif isinstance(policy, dict):
        sorted_keys = list(policy.keys())
        sorted_keys.sort()
        for key in sorted_keys:
            element = _canonify_policy_dict_item(policy[key], key)
            hashed_policy = _hashable_policy(element, [])
            tupleified = _tuplify_list(hashed_policy)
            policy_list.append((key, tupleified))
    if len(policy_list) == 1 and isinstance(policy_list[0], tuple):
        policy_list = policy_list[0]
    if isinstance(policy_list, list):
        policy_list.sort(key=cmp_to_key(_py3cmp))
    return policy_list