from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InstanceGroupManagersService(base_api.BaseApiService):
    """Service class for the instanceGroupManagers resource."""
    _NAME = 'instanceGroupManagers'

    def __init__(self, client):
        super(ComputeBeta.InstanceGroupManagersService, self).__init__(client)
        self._upload_configs = {}

    def AbandonInstances(self, request, global_params=None):
        """Flags the specified instances to be removed from the managed instance group. Abandoning an instance does not delete the instance, but it does remove the instance from any target pools that are applied by the managed instance group. This method reduces the targetSize of the managed instance group by the number of instances that you abandon. This operation is marked as DONE when the action is scheduled even if the instances have not yet been removed from the group. You must separately verify the status of the abandoning action with the listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersAbandonInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AbandonInstances')
        return self._RunMethod(config, request, global_params=global_params)
    AbandonInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.abandonInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/abandonInstances', request_field='instanceGroupManagersAbandonInstancesRequest', request_type_name='ComputeInstanceGroupManagersAbandonInstancesRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of managed instance groups and groups them by zone. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeInstanceGroupManagersAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagerAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagers.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/instanceGroupManagers', request_field='', request_type_name='ComputeInstanceGroupManagersAggregatedListRequest', response_type_name='InstanceGroupManagerAggregatedList', supports_download=False)

    def ApplyUpdatesToInstances(self, request, global_params=None):
        """Applies changes to selected instances on the managed instance group. This method can be used to apply new overrides and/or new versions.

      Args:
        request: (ComputeInstanceGroupManagersApplyUpdatesToInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ApplyUpdatesToInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ApplyUpdatesToInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.applyUpdatesToInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/applyUpdatesToInstances', request_field='instanceGroupManagersApplyUpdatesRequest', request_type_name='ComputeInstanceGroupManagersApplyUpdatesToInstancesRequest', response_type_name='Operation', supports_download=False)

    def CreateInstances(self, request, global_params=None):
        """Creates instances with per-instance configurations in this managed instance group. Instances are created using the current instance template. The create instances operation is marked DONE if the createInstances request is successful. The underlying actions take additional time. You must separately verify the status of the creating or actions with the listmanagedinstances method.

      Args:
        request: (ComputeInstanceGroupManagersCreateInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateInstances')
        return self._RunMethod(config, request, global_params=global_params)
    CreateInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.createInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/createInstances', request_field='instanceGroupManagersCreateInstancesRequest', request_type_name='ComputeInstanceGroupManagersCreateInstancesRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified managed instance group and all of the instances in that group. Note that the instance group must not belong to a backend service. Read Deleting an instance group for more information.

      Args:
        request: (ComputeInstanceGroupManagersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.instanceGroupManagers.delete', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}', request_field='', request_type_name='ComputeInstanceGroupManagersDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group for immediate deletion. The instances are also removed from any target pools of which they were a member. This method reduces the targetSize of the managed instance group by the number of instances that you delete. This operation is marked as DONE when the action is scheduled even if the instances are still being deleted. You must separately verify the status of the deleting action with the listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersDeleteInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteInstances')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.deleteInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deleteInstances', request_field='instanceGroupManagersDeleteInstancesRequest', request_type_name='ComputeInstanceGroupManagersDeleteInstancesRequest', response_type_name='Operation', supports_download=False)

    def DeletePerInstanceConfigs(self, request, global_params=None):
        """Deletes selected per-instance configurations for the managed instance group.

      Args:
        request: (ComputeInstanceGroupManagersDeletePerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeletePerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    DeletePerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.deletePerInstanceConfigs', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deletePerInstanceConfigs', request_field='instanceGroupManagersDeletePerInstanceConfigsReq', request_type_name='ComputeInstanceGroupManagersDeletePerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns all of the details about the specified managed instance group.

      Args:
        request: (ComputeInstanceGroupManagersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManager) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagers.get', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}', request_field='', request_type_name='ComputeInstanceGroupManagersGetRequest', response_type_name='InstanceGroupManager', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a managed instance group using the information that you specify in the request. After the group is created, instances in the group are created using the specified instance template. This operation is marked as DONE when the group is created even if the instances in the group have not yet been created. You must separately verify the status of the individual instances with the listmanagedinstances method. A managed instance group can have up to 1000 VM instances per group. Please contact Cloud Support if you need an increase in this limit.

      Args:
        request: (ComputeInstanceGroupManagersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers', request_field='instanceGroupManager', request_type_name='ComputeInstanceGroupManagersInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of managed instance groups that are contained within the specified project and zone.

      Args:
        request: (ComputeInstanceGroupManagersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagerList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagers.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers', request_field='', request_type_name='ComputeInstanceGroupManagersListRequest', response_type_name='InstanceGroupManagerList', supports_download=False)

    def ListErrors(self, request, global_params=None):
        """Lists all errors thrown by actions on instances for a given managed instance group. The filter and orderBy query parameters are not supported.

      Args:
        request: (ComputeInstanceGroupManagersListErrorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagersListErrorsResponse) The response message.
      """
        config = self.GetMethodConfig('ListErrors')
        return self._RunMethod(config, request, global_params=global_params)
    ListErrors.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagers.listErrors', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/listErrors', request_field='', request_type_name='ComputeInstanceGroupManagersListErrorsRequest', response_type_name='InstanceGroupManagersListErrorsResponse', supports_download=False)

    def ListManagedInstances(self, request, global_params=None):
        """Lists all of the instances in the managed instance group. Each instance in the list has a currentAction, which indicates the action that the managed instance group is performing on the instance. For example, if the group is still creating an instance, the currentAction is CREATING. If a previous action failed, the list displays the errors for that failed action. The orderBy query parameter is not supported. The `pageToken` query parameter is supported only if the group's `listManagedInstancesResults` field is set to `PAGINATED`.

      Args:
        request: (ComputeInstanceGroupManagersListManagedInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagersListManagedInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('ListManagedInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ListManagedInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.listManagedInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/listManagedInstances', request_field='', request_type_name='ComputeInstanceGroupManagersListManagedInstancesRequest', response_type_name='InstanceGroupManagersListManagedInstancesResponse', supports_download=False)

    def ListPerInstanceConfigs(self, request, global_params=None):
        """Lists all of the per-instance configurations defined for the managed instance group. The orderBy query parameter is not supported.

      Args:
        request: (ComputeInstanceGroupManagersListPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagersListPerInstanceConfigsResp) The response message.
      """
        config = self.GetMethodConfig('ListPerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    ListPerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.listPerInstanceConfigs', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/listPerInstanceConfigs', request_field='', request_type_name='ComputeInstanceGroupManagersListPerInstanceConfigsRequest', response_type_name='InstanceGroupManagersListPerInstanceConfigsResp', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a managed instance group using the information that you specify in the request. This operation is marked as DONE when the group is patched even if the instances in the group are still in the process of being patched. You must separately verify the status of the individual instances with the listManagedInstances method. This method supports PATCH semantics and uses the JSON merge patch format and processing rules. If you update your group to specify a new template or instance configuration, it's possible that your intended specification for each VM in the group is different from the current state of that VM. To learn how to apply an updated configuration to the VMs in a MIG, see Updating instances in a MIG.

      Args:
        request: (ComputeInstanceGroupManagersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.instanceGroupManagers.patch', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}', request_field='instanceGroupManagerResource', request_type_name='ComputeInstanceGroupManagersPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchPerInstanceConfigs(self, request, global_params=None):
        """Inserts or patches per-instance configurations for the managed instance group. perInstanceConfig.name serves as a key used to distinguish whether to perform insert or patch.

      Args:
        request: (ComputeInstanceGroupManagersPatchPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchPerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    PatchPerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.patchPerInstanceConfigs', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/patchPerInstanceConfigs', request_field='instanceGroupManagersPatchPerInstanceConfigsReq', request_type_name='ComputeInstanceGroupManagersPatchPerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)

    def RecreateInstances(self, request, global_params=None):
        """Flags the specified VM instances in the managed instance group to be immediately recreated. Each instance is recreated using the group's current configuration. This operation is marked as DONE when the flag is set even if the instances have not yet been recreated. You must separately verify the status of each instance by checking its currentAction field; for more information, see Checking the status of managed instances. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersRecreateInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RecreateInstances')
        return self._RunMethod(config, request, global_params=global_params)
    RecreateInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.recreateInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/recreateInstances', request_field='instanceGroupManagersRecreateInstancesRequest', request_type_name='ComputeInstanceGroupManagersRecreateInstancesRequest', response_type_name='Operation', supports_download=False)

    def Resize(self, request, global_params=None):
        """Resizes the managed instance group. If you increase the size, the group creates new instances using the current instance template. If you decrease the size, the group deletes instances. The resize operation is marked DONE when the resize actions are scheduled even if the group has not yet added or deleted any instances. You must separately verify the status of the creating or deleting actions with the listmanagedinstances method. When resizing down, the instance group arbitrarily chooses the order in which VMs are deleted. The group takes into account some VM attributes when making the selection including: + The status of the VM instance. + The health of the VM instance. + The instance template version the VM is based on. + For regional managed instance groups, the location of the VM instance. This list is subject to change. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted.

      Args:
        request: (ComputeInstanceGroupManagersResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.resize', ordered_params=['project', 'zone', 'instanceGroupManager', 'size'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId', 'size'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resize', request_field='', request_type_name='ComputeInstanceGroupManagersResizeRequest', response_type_name='Operation', supports_download=False)

    def ResizeAdvanced(self, request, global_params=None):
        """Resizes the managed instance group with advanced configuration options like disabling creation retries. This is an extended version of the resize method. If you increase the size of the instance group, the group creates new instances using the current instance template. If you decrease the size, the group deletes instances. The resize operation is marked DONE when the resize actions are scheduled even if the group has not yet added or deleted any instances. You must separately verify the status of the creating, creatingWithoutRetries, or deleting actions with the get or listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted.

      Args:
        request: (ComputeInstanceGroupManagersResizeAdvancedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResizeAdvanced')
        return self._RunMethod(config, request, global_params=global_params)
    ResizeAdvanced.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.resizeAdvanced', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeAdvanced', request_field='instanceGroupManagersResizeAdvancedRequest', request_type_name='ComputeInstanceGroupManagersResizeAdvancedRequest', response_type_name='Operation', supports_download=False)

    def ResumeInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be resumed. This method increases the targetSize and decreases the targetSuspendedSize of the managed instance group by the number of instances that you resume. The resumeInstances operation is marked DONE if the resumeInstances request is successful. The underlying actions take additional time. You must separately verify the status of the RESUMING action with the listmanagedinstances method. In this request, you can only specify instances that are suspended. For example, if an instance was previously suspended using the suspendInstances method, it can be resumed using the resumeInstances method. If a health check is attached to the managed instance group, the specified instances will be verified as healthy after they are resumed. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersResumeInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResumeInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ResumeInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.resumeInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resumeInstances', request_field='instanceGroupManagersResumeInstancesRequest', request_type_name='ComputeInstanceGroupManagersResumeInstancesRequest', response_type_name='Operation', supports_download=False)

    def SetAutoHealingPolicies(self, request, global_params=None):
        """Motifies the autohealing policy for the instances in this managed instance group. [Deprecated] This method is deprecated. Use instanceGroupManagers.patch instead.

      Args:
        request: (ComputeInstanceGroupManagersSetAutoHealingPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetAutoHealingPolicies')
        return self._RunMethod(config, request, global_params=global_params)
    SetAutoHealingPolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.setAutoHealingPolicies', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setAutoHealingPolicies', request_field='instanceGroupManagersSetAutoHealingRequest', request_type_name='ComputeInstanceGroupManagersSetAutoHealingPoliciesRequest', response_type_name='Operation', supports_download=False)

    def SetInstanceTemplate(self, request, global_params=None):
        """Specifies the instance template to use when creating new instances in this group. The templates for existing instances in the group do not change unless you run recreateInstances, run applyUpdatesToInstances, or set the group's updatePolicy.type to PROACTIVE.

      Args:
        request: (ComputeInstanceGroupManagersSetInstanceTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetInstanceTemplate')
        return self._RunMethod(config, request, global_params=global_params)
    SetInstanceTemplate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.setInstanceTemplate', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate', request_field='instanceGroupManagersSetInstanceTemplateRequest', request_type_name='ComputeInstanceGroupManagersSetInstanceTemplateRequest', response_type_name='Operation', supports_download=False)

    def SetTargetPools(self, request, global_params=None):
        """Modifies the target pools to which all instances in this managed instance group are assigned. The target pools automatically apply to all of the instances in the managed instance group. This operation is marked DONE when you make the request even if the instances have not yet been added to their target pools. The change might take some time to apply to all of the instances in the group depending on the size of the group.

      Args:
        request: (ComputeInstanceGroupManagersSetTargetPoolsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetTargetPools')
        return self._RunMethod(config, request, global_params=global_params)
    SetTargetPools.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.setTargetPools', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setTargetPools', request_field='instanceGroupManagersSetTargetPoolsRequest', request_type_name='ComputeInstanceGroupManagersSetTargetPoolsRequest', response_type_name='Operation', supports_download=False)

    def StartInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be started. This method increases the targetSize and decreases the targetStoppedSize of the managed instance group by the number of instances that you start. The startInstances operation is marked DONE if the startInstances request is successful. The underlying actions take additional time. You must separately verify the status of the STARTING action with the listmanagedinstances method. In this request, you can only specify instances that are stopped. For example, if an instance was previously stopped using the stopInstances method, it can be started using the startInstances method. If a health check is attached to the managed instance group, the specified instances will be verified as healthy after they are started. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersStartInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartInstances')
        return self._RunMethod(config, request, global_params=global_params)
    StartInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.startInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/startInstances', request_field='instanceGroupManagersStartInstancesRequest', request_type_name='ComputeInstanceGroupManagersStartInstancesRequest', response_type_name='Operation', supports_download=False)

    def StopInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be immediately stopped. You can only specify instances that are running in this request. This method reduces the targetSize and increases the targetStoppedSize of the managed instance group by the number of instances that you stop. The stopInstances operation is marked DONE if the stopInstances request is successful. The underlying actions take additional time. You must separately verify the status of the STOPPING action with the listmanagedinstances method. If the standbyPolicy.initialDelaySec field is set, the group delays stopping the instances until initialDelaySec have passed from instance.creationTimestamp (that is, when the instance was created). This delay gives your application time to set itself up and initialize on the instance. If more than initialDelaySec seconds have passed since instance.creationTimestamp when this method is called, there will be zero delay. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is stopped. Stopped instances can be started using the startInstances method. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersStopInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopInstances')
        return self._RunMethod(config, request, global_params=global_params)
    StopInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.stopInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/stopInstances', request_field='instanceGroupManagersStopInstancesRequest', request_type_name='ComputeInstanceGroupManagersStopInstancesRequest', response_type_name='Operation', supports_download=False)

    def SuspendInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be immediately suspended. You can only specify instances that are running in this request. This method reduces the targetSize and increases the targetSuspendedSize of the managed instance group by the number of instances that you suspend. The suspendInstances operation is marked DONE if the suspendInstances request is successful. The underlying actions take additional time. You must separately verify the status of the SUSPENDING action with the listmanagedinstances method. If the standbyPolicy.initialDelaySec field is set, the group delays suspension of the instances until initialDelaySec have passed from instance.creationTimestamp (that is, when the instance was created). This delay gives your application time to set itself up and initialize on the instance. If more than initialDelaySec seconds have passed since instance.creationTimestamp when this method is called, there will be zero delay. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is suspended. Suspended instances can be resumed using the resumeInstances method. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeInstanceGroupManagersSuspendInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SuspendInstances')
        return self._RunMethod(config, request, global_params=global_params)
    SuspendInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.suspendInstances', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/suspendInstances', request_field='instanceGroupManagersSuspendInstancesRequest', request_type_name='ComputeInstanceGroupManagersSuspendInstancesRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeInstanceGroupManagersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeInstanceGroupManagersTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a managed instance group using the information that you specify in the request. This operation is marked as DONE when the group is updated even if the instances in the group have not yet been updated. You must separately verify the status of the individual instances with the listManagedInstances method. If you update your group to specify a new template or instance configuration, it's possible that your intended specification for each VM in the group is different from the current state of that VM. To learn how to apply an updated configuration to the VMs in a MIG, see Updating instances in a MIG.

      Args:
        request: (ComputeInstanceGroupManagersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.instanceGroupManagers.update', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}', request_field='instanceGroupManagerResource', request_type_name='ComputeInstanceGroupManagersUpdateRequest', response_type_name='Operation', supports_download=False)

    def UpdatePerInstanceConfigs(self, request, global_params=None):
        """Inserts or updates per-instance configurations for the managed instance group. perInstanceConfig.name serves as a key used to distinguish whether to perform insert or patch.

      Args:
        request: (ComputeInstanceGroupManagersUpdatePerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdatePerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    UpdatePerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagers.updatePerInstanceConfigs', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/updatePerInstanceConfigs', request_field='instanceGroupManagersUpdatePerInstanceConfigsReq', request_type_name='ComputeInstanceGroupManagersUpdatePerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)