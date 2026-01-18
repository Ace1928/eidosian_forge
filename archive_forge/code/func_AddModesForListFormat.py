from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddModesForListFormat(resource):
    return dict(resource, x_gcloud_subnet_mode=GetSubnetMode(resource), x_gcloud_bgp_routing_mode=GetBgpRoutingMode(resource))