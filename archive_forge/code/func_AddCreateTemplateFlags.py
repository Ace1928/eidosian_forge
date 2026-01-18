from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
def AddCreateTemplateFlags(parser):
    """Add a list of flags for create job templates."""
    template_config = parser.add_mutually_exclusive_group(required=True)
    template_config.add_argument('--json', help='Job template in json format.')
    template_config.add_argument('--file', help='Path to job template.')