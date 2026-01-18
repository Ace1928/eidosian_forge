from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddExportCustomRoutesFlag(parser):
    """Adds exportCustomRoutes flag to the argparse.ArgumentParser."""
    parser.add_argument('--export-custom-routes', action='store_true', default=None, help='        If set, the network will export custom routes to peer network. Use\n        --no-export-custom-routes to disable it.\n      ')