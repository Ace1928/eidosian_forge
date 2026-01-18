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
class AddIamPolicyBinding(_IamPolicyBindingCmd):
    usage = 'add-iam-policy-binding --member=<member> --role=<role> [(-d|-t)] <identifier>'

    def __init__(self, name: str, fv: flags.FlagValues):
        super(AddIamPolicyBinding, self).__init__(name, fv, verb='Add binding to')
        self._ProcessCommandRc(fv)

    def RunWithArgs(self, identifier: str) -> Optional[int]:
        """Add a binding to a BigQuery resource's policy in IAM.

    Usage:
      add-iam-policy-binding --member=<member> --role=<role> <identifier>

    One binding consists of a member and a role, which are specified with
    (required) flags.

    Examples:

      bq add-iam-policy-binding \\
        --member='user:myaccount@gmail.com' \\
        --role='roles/bigquery.dataViewer' \\
        table1

      bq add-iam-policy-binding \\
        --member='serviceAccount:my.service.account@my-domain.com' \\
        --role='roles/bigquery.dataEditor' \\
        project1:dataset1.table1

      bq add-iam-policy-binding \\
       --member='allAuthenticatedUsers' \\
       --role='roles/bigquery.dataViewer' \\
       --project_id=proj -t ds.table1

    Arguments:
      identifier: The identifier of the resource. Presently only table and view
        resources are fully supported. (Last updated: 2020-08-03)
    """
        client = bq_cached_client.Client.Get()
        reference = self.GetReferenceFromIdentifier(client, identifier)
        policy = self.GetPolicyForReference(client, reference)
        if 'etag' not in [key.lower() for key in policy]:
            raise ValueError("Policy doesn't have an 'etag' field. This is unexpected. The etag is required to prevent unexpected results from concurrent edits.")
        self.AddBindingToPolicy(policy, self.member, self.role)
        result_policy = self.SetPolicyForReference(client, reference, policy)
        print("Successfully added member '{member}' to role '{role}' in IAM policy for {resource_type} '{identifier}':\n".format(member=self.member, role=self.role, resource_type=reference.typename, identifier=reference))
        bq_utils.PrintFormattedJsonObject(result_policy, default_format='prettyjson')

    @staticmethod
    def AddBindingToPolicy(policy, member, role):
        """Add a binding to an IAM policy.

    Args:
      policy: The policy object, composed of dictionaries, lists, and primitive
        types. This object will be modified, and also returned for convenience.
      member: The string to insert into the 'members' array of the binding.
      role: The role string of the binding to remove.

    Returns:
      The same object referenced by the policy arg, after adding the binding.
    """
        if policy.get('version', 1) > 1:
            raise ValueError('Only policy versions up to 1 are supported. version: {version}'.format(version=policy.get('version', 'None')))
        bindings = policy.setdefault('bindings', [])
        if not isinstance(bindings, list):
            raise ValueError("Policy field 'bindings' does not have an array-type value. 'bindings': {value}".format(value=repr(bindings)))
        for binding in bindings:
            if not isinstance(binding, dict):
                raise ValueError("At least one element of the policy's 'bindings' array is not an object type. element: {value}".format(value=repr(binding)))
            if binding.get('role') == role:
                break
        else:
            binding = {'role': role}
            bindings.append(binding)
        members = binding.setdefault('members', [])
        if not isinstance(members, list):
            raise ValueError("Policy binding field 'members' does not have an array-type value. 'members': {value}".format(value=repr(members)))
        if member not in members:
            members.append(member)
        return policy