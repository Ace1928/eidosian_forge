from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List, Optional
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import identity_pool_waiter
from googlecloudsdk.core import resources as sdkresources
def CreateGcpWorkloadSource(client, messages, workload_source_id: str, resources: Optional[List[str]], attached_service_accounts: Optional[List[str]], parent: str, for_managed_identity: bool=False):
    """Make API calls to Create a GCP workload source.

  Args:
    client: the iam v1 client.
    messages: the iam v1 messages.
    workload_source_id: the workload source id to be created.
    resources: the list of resource attribute conditions to be created
    attached_service_accounts: the list of service account attribute conditions
      to be created
    parent: the parent resource name, should be a namespace or a managed
      identity resource
    for_managed_identity: whether to create the workload source under a managed
      identity

  Returns:
    The LRO ref for a create response
  """
    conditions = []
    if resources is not None:
        conditions += [messages.WorkloadSourceCondition(attribute='resource', value=resource) for resource in resources]
    if attached_service_accounts is not None:
        conditions += [messages.WorkloadSourceCondition(attribute='attached_service_account', value=account) for account in attached_service_accounts]
    new_workload_source = messages.WorkloadSource(conditionSet=messages.WorkloadSourceConditionSet(conditions=conditions))
    if for_managed_identity:
        return client.projects_locations_workloadIdentityPools_namespaces_managedIdentities_workloadSources.Create(messages.IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesCreateRequest(parent=parent, workloadSource=new_workload_source, workloadSourceId=workload_source_id))
    else:
        return client.projects_locations_workloadIdentityPools_namespaces_workloadSources.Create(messages.IamProjectsLocationsWorkloadIdentityPoolsNamespacesWorkloadSourcesCreateRequest(parent=parent, workloadSource=new_workload_source, workloadSourceId=workload_source_id))