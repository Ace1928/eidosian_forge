from openstack import resource
class RecoveryWorkflowDetailItem(resource.Resource):
    progress = resource.Body('progress')
    name = resource.Body('name')
    state = resource.Body('state')
    progress_details = resource.Body('progress_details', type=list, list_type=ProgressDetailsItem)