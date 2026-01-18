from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
from osc_lib import utils
from neutronclient._i18n import _
def _find_identity_resource(identity_client_manager, name_or_id, resource_type, **kwargs):
    """Find a specific identity resource.

    Using keystoneclient's manager, attempt to find a specific resource by its
    name or ID. If Forbidden to find the resource (a common case if the user
    does not have permission), then return the resource by creating a local
    instance of keystoneclient's Resource.

    The parameter identity_client_manager is a keystoneclient manager,
    for example: keystoneclient.v3.users or keystoneclient.v3.projects.

    The parameter resource_type is a keystoneclient resource, for example:
    keystoneclient.v3.users.User or keystoneclient.v3.projects.Project.

    :param identity_client_manager: the manager that contains the resource
    :type identity_client_manager: `keystoneclient.base.CrudManager`
    :param name_or_id: the resources's name or ID
    :type name_or_id: string
    :param resource_type: class that represents the resource type
    :type resource_type: `keystoneclient.base.Resource`

    :returns: the resource in question
    :rtype: `keystoneclient.base.Resource`

    """
    try:
        identity_resource = utils.find_resource(identity_client_manager, name_or_id, **kwargs)
        if identity_resource is not None:
            return identity_resource
    except identity_exc.Forbidden:
        pass
    return resource_type(None, {'id': name_or_id, 'name': name_or_id})