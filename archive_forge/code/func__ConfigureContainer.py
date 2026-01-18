from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
def _ConfigureContainer(self, version, **kwargs):
    """Adds `container` and `routes` fields to version."""
    if not any(kwargs.values()):
        return
    if not kwargs['image']:
        set_flags = ', '.join(['--{}'.format(k) for k, v in sorted(kwargs.items()) if v])
        raise ValueError('--image was not provided, but other container related flags were specified. Please specify --image or remove the following flags: {}'.format(set_flags))
    version.container = self.messages.GoogleCloudMlV1ContainerSpec(image=kwargs['image'])
    if kwargs['command']:
        version.container.command = kwargs['command']
    if kwargs['args']:
        version.container.args = kwargs['args']
    if kwargs['env_vars']:
        version.container.env = [self.messages.GoogleCloudMlV1EnvVar(name=name, value=value) for name, value in kwargs['env_vars'].items()]
    if kwargs['ports']:
        version.container.ports = [self.messages.GoogleCloudMlV1ContainerPort(containerPort=p) for p in kwargs['ports']]
    if kwargs['predict_route'] or kwargs['health_route']:
        version.routes = self.messages.GoogleCloudMlV1RouteMap(predict=kwargs['predict_route'], health=kwargs['health_route'])