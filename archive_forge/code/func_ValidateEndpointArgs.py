from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
def ValidateEndpointArgs(network=None, public_endpoint_enabled=None):
    """Validates the network and public_endpoint_enabled."""
    if network is not None and public_endpoint_enabled:
        raise exceptions.InvalidArgumentException('Please either set --network for private endpoint, or set --public-endpoint-enabled', 'for public enpdoint.')