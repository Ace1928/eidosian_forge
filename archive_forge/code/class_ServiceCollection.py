import copy
from docker.errors import create_unexpected_kwargs_error, InvalidArgument
from docker.types import TaskTemplate, ContainerSpec, Placement, ServiceMode
from .resource import Model, Collection
class ServiceCollection(Collection):
    """Services on the Docker server."""
    model = Service

    def create(self, image, command=None, **kwargs):
        """
        Create a service. Similar to the ``docker service create`` command.

        Args:
            image (str): The image name to use for the containers.
            command (list of str or str): Command to run.
            args (list of str): Arguments to the command.
            constraints (list of str): :py:class:`~docker.types.Placement`
                constraints.
            preferences (list of tuple): :py:class:`~docker.types.Placement`
                preferences.
            maxreplicas (int): :py:class:`~docker.types.Placement` maxreplicas
                or (int) representing maximum number of replicas per node.
            platforms (list of tuple): A list of platform constraints
                expressed as ``(arch, os)`` tuples.
            container_labels (dict): Labels to apply to the container.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.
            env (list of str): Environment variables, in the form
                ``KEY=val``.
            hostname (string): Hostname to set on the container.
            init (boolean): Run an init inside the container that forwards
                signals and reaps processes
            isolation (string): Isolation technology used by the service's
                containers. Only used for Windows containers.
            labels (dict): Labels to apply to the service.
            log_driver (str): Log driver to use for containers.
            log_driver_options (dict): Log driver options.
            mode (ServiceMode): Scheduling mode for the service.
                Default:``None``
            mounts (list of str): Mounts for the containers, in the form
                ``source:target:options``, where options is either
                ``ro`` or ``rw``.
            name (str): Name to give to the service.
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            resources (Resources): Resource limits and reservations.
            restart_policy (RestartPolicy): Restart policy for containers.
            secrets (list of :py:class:`~docker.types.SecretReference`): List
                of secrets accessible to containers for this service.
            stop_grace_period (int): Amount of time to wait for
                containers to terminate before forcefully killing them.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            user (str): User to run commands as.
            workdir (str): Working directory for commands to run.
            tty (boolean): Whether a pseudo-TTY should be allocated.
            groups (:py:class:`list`): A list of additional groups that the
                container process will run as.
            open_stdin (boolean): Open ``stdin``
            read_only (boolean): Mount the container's root filesystem as read
                only.
            stop_signal (string): Set signal to stop the service's containers
            healthcheck (Healthcheck): Healthcheck
                configuration for this service.
            hosts (:py:class:`dict`): A set of host to IP mappings to add to
                the container's `hosts` file.
            dns_config (DNSConfig): Specification for DNS
                related configurations in resolver configuration file.
            configs (:py:class:`list`): List of
                :py:class:`~docker.types.ConfigReference` that will be exposed
                to the service.
            privileges (Privileges): Security options for the service's
                containers.
            cap_add (:py:class:`list`): A list of kernel capabilities to add to
                the default set for the container.
            cap_drop (:py:class:`list`): A list of kernel capabilities to drop
                from the default set for the container.
            sysctls (:py:class:`dict`): A dict of sysctl values to add to the
                container

        Returns:
            :py:class:`Service`: The created service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        kwargs['image'] = image
        kwargs['command'] = command
        create_kwargs = _get_create_service_kwargs('create', kwargs)
        service_id = self.client.api.create_service(**create_kwargs)
        return self.get(service_id)

    def get(self, service_id, insert_defaults=None):
        """
        Get a service.

        Args:
            service_id (str): The ID of the service.
            insert_defaults (boolean): If true, default values will be merged
                into the output.

        Returns:
            :py:class:`Service`: The service.

        Raises:
            :py:class:`docker.errors.NotFound`
                If the service does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
            :py:class:`docker.errors.InvalidVersion`
                If one of the arguments is not supported with the current
                API version.
        """
        return self.prepare_model(self.client.api.inspect_service(service_id, insert_defaults))

    def list(self, **kwargs):
        """
        List services.

        Args:
            filters (dict): Filters to process on the nodes list. Valid
                filters: ``id``, ``name`` , ``label`` and ``mode``.
                Default: ``None``.
            status (bool): Include the service task count of running and
                desired tasks. Default: ``None``.

        Returns:
            list of :py:class:`Service`: The services.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return [self.prepare_model(s) for s in self.client.api.services(**kwargs)]