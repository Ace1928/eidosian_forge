from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class ContainerResource(k8s_object.KubernetesObject):
    """Wraps a resource message with a container, making fields more convenient.

  Provides convience fields for Cloud Run resources that contain a container.
  These resources also typically have other overlapping fields such as volumes
  which are also handled by this wrapper.
  """

    @property
    def env_vars(self):
        """Returns a mutable, dict-like object to manage env vars.

    The returned object can be used like a dictionary, and any modifications to
    the returned object (i.e. setting and deleting keys) modify the underlying
    nested env vars fields.
    """
        return self.container.env_vars

    @property
    def image(self):
        """URL to container."""
        return self.container.image

    @image.setter
    def image(self, value):
        self.container.image = value

    @property
    def command(self):
        """command to be invoked by container."""
        return self.container.command

    @property
    def container(self):
        """The container in the revisionTemplate."""
        containers = self.containers.values()
        if not containers:
            return self.containers['']
        if len(containers) == 1:
            return next(iter(containers))
        for container in containers:
            if container.ports:
                return container
        raise ValueError('missing ingress container')

    @property
    def containers(self):
        """The containers in the revisionTemplate."""
        return ContainersAsDictionaryWrapper(self.spec.containers, self.volumes, self._messages)

    @property
    def resource_limits(self):
        """The resource limits as a dictionary { resource name: limit}."""
        return self.container.resource_limits

    @property
    def volumes(self):
        """Returns a dict-like object to manage volumes.

    There are additional properties on the object (e.g. `.secrets`) that can
    be used to access a mutable, dict-like object for managing volumes of a
    given type. Any modifications to the returned object for these properties
    (i.e. setting and deleting keys) modify the underlying nested volumes.
    """
        return VolumesAsDictionaryWrapper(self.spec.volumes, self._messages.Volume)

    @property
    def dependencies(self) -> Mapping[str, Sequence[str]]:
        """Returns a dictionary of container dependencies.

    Container dependencies are stored in the
    'run.googleapis.com/container-dependencies' annotation. The returned
    dictionary maps containers to a list of their dependencies by name. Note
    that updates to the returned dictionary do not update the resource's
    container dependencies unless the dependencies setter is used.
    """
        dependencies = {}
        if k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION in self.annotations:
            dependencies = json.loads(self.annotations[k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION])
        return dependencies

    @dependencies.setter
    def dependencies(self, dependencies: Mapping[str, Sequence[str]]):
        """Sets the resource's container dependencies.

    Args:
      dependencies: A dictionary mapping containers to a list of their
        dependencies by name.

    Container dependencies are stored in the
    'run.googleapis.com/container-dependencies' annotation as json. Setting an
    empty set of dependencies will clear this annotation.
    """
        if dependencies:
            self.annotations[k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION] = json.dumps({k: list(v) for k, v in dependencies.items()})
        elif k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION in self.annotations:
            del self.annotations[k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION]