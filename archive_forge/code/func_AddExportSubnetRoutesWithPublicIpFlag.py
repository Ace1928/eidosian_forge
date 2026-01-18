from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddExportSubnetRoutesWithPublicIpFlag(parser):
    """Adds exportSubnetRoutesWithPublicIp flag to the argparse.ArgumentParser."""
    parser.add_argument('--export-subnet-routes-with-public-ip', action='store_true', default=None, help='        If set, the network will export subnet routes with addresses in the\n        public IP ranges to peer network.\n        Use --no-export-subnet-routes-with-public-ip to disable it.\n      ')