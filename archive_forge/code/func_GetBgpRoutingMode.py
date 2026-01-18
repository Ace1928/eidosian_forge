from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetBgpRoutingMode(network):
    """Returns the BGP routing mode of the input network."""
    return network.get('routingConfig', {}).get('routingMode')