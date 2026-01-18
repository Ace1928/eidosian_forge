from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from typing import Optional
from absl import app
from absl import flags
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_id_utils
@staticmethod
def RemoveBindingFromPolicy(policy, member, role):
    """Remove a binding from an IAM policy.

    Will remove the member from the binding, and remove the entire binding if
    its members array is empty.

    Args:
      policy: The policy object, composed of dictionaries, lists, and primitive
        types. This object will be modified, and also returned for convenience.
      member: The string to remove from the 'members' array of the binding.
      role: The role string of the binding to remove.

    Returns:
      The same object referenced by the policy arg, after adding the binding.
    """
    if policy.get('version', 1) > 1:
        raise ValueError('Only policy versions up to 1 are supported. version: {version}'.format(version=policy.get('version', 'None')))
    bindings = policy.get('bindings', [])
    if not isinstance(bindings, list):
        raise ValueError("Policy field 'bindings' does not have an array-type value. 'bindings': {value}".format(value=repr(bindings)))
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("At least one element of the policy's 'bindings' array is not an object type. element: {value}".format(value=repr(binding)))
        if binding.get('role') == role:
            members = binding.get('members', [])
            if not isinstance(members, list):
                raise ValueError("Policy binding field 'members' does not have an array-type value. 'members': {value}".format(value=repr(members)))
            for j, member_j in enumerate(members):
                if member_j == member:
                    del members[j]
                    bindings = [b for b in bindings if b.get('members', [])]
                    policy['bindings'] = bindings
                    return policy
    raise app.UsageError("No binding found for member '{member}' in role '{role}'".format(member=member, role=role))