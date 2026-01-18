import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def add_endpoint_group_to_project(self, endpoint_group_id, project_id):
    """Add an endpoint group to project association.

        :param endpoint_group_id: identity of endpoint to associate
        :type endpoint_group_id: string
        :param project_id: identity of project to associate
        :type project_id: string
        :raises keystone.exception.Conflict: If the endpoint group was already
            added to the project.
        :returns: None.

        """
    raise exception.NotImplemented()