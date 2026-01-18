from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddRequiredConfigFlag(parser):
    parser.add_argument('--config-name', help='The name of the configuration resource to use.', required=True)