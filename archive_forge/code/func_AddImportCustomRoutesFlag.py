from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddImportCustomRoutesFlag(parser):
    """Adds importCustomRoutes flag to the argparse.ArgumentParser."""
    parser.add_argument('--import-custom-routes', action='store_true', default=None, help='        If set, the network will import custom routes from peer network. Use\n        --no-import-custom-routes to disable it.\n      ')