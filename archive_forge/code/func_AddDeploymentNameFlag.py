from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDeploymentNameFlag(parser):
    parser.add_argument('NAME', nargs=1, help='Name of the locally deployed Google Cloud function.')