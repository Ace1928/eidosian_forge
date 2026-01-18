from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class TaskTemplate(dict):
    """
    Describe the task specification to be used when creating or updating a
    service.

    Args:

        container_spec (ContainerSpec): Container settings for containers
          started as part of this task.
        log_driver (DriverConfig): Log configuration for containers created as
          part of the service.
        resources (Resources): Resource requirements which apply to each
          individual container created as part of the service.
        restart_policy (RestartPolicy): Specification for the restart policy
          which applies to containers created as part of this service.
        placement (Placement): Placement instructions for the scheduler.
            If a list is passed instead, it is assumed to be a list of
            constraints as part of a :py:class:`Placement` object.
        networks (:py:class:`list`): List of network names or IDs or
            :py:class:`NetworkAttachmentConfig` to attach the service to.
        force_update (int): A counter that triggers an update even if no
            relevant parameters have been changed.
    """

    def __init__(self, container_spec, resources=None, restart_policy=None, placement=None, log_driver=None, networks=None, force_update=None):
        self['ContainerSpec'] = container_spec
        if resources:
            self['Resources'] = resources
        if restart_policy:
            self['RestartPolicy'] = restart_policy
        if placement:
            if isinstance(placement, list):
                placement = Placement(constraints=placement)
            self['Placement'] = placement
        if log_driver:
            self['LogDriver'] = log_driver
        if networks:
            self['Networks'] = convert_service_networks(networks)
        if force_update is not None:
            if not isinstance(force_update, int):
                raise TypeError('force_update must be an integer')
            self['ForceUpdate'] = force_update

    @property
    def container_spec(self):
        return self.get('ContainerSpec')

    @property
    def resources(self):
        return self.get('Resources')

    @property
    def restart_policy(self):
        return self.get('RestartPolicy')

    @property
    def placement(self):
        return self.get('Placement')