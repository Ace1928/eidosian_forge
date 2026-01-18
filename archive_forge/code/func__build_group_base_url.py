from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoint_groups
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import projects
def _build_group_base_url(self, project=None, endpoint_group=None):
    project_id = base.getid(project)
    endpoint_group_id = base.getid(endpoint_group)
    if project_id and endpoint_group_id:
        api_path = '/endpoint_groups/%s/projects/%s' % (endpoint_group_id, project_id)
    elif project_id:
        api_path = '/projects/%s/endpoint_groups' % project_id
    elif endpoint_group_id:
        api_path = '/endpoint_groups/%s/projects' % endpoint_group_id
    else:
        msg = _('Must specify a project, an endpoint group, or both')
        raise exceptions.ValidationError(msg)
    return self.OS_EP_FILTER_EXT + api_path