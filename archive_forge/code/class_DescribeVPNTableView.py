from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class DescribeVPNTableView:
    """View model for VPN connections describe."""

    def __init__(self, name, create_time, cluster, vpc, state, error):
        self.name = name
        self.create_time = create_time
        self.cluster = cluster
        self.vpc = vpc
        self.state = state
        if error:
            self.error = error