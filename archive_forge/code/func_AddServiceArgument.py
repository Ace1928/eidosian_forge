from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddServiceArgument(parser):
    base.ChoiceArgument('--service', required=True, metavar='SERVICE_NAME', choices=['container-threat-detection', 'event-threat-detection', 'rapid-vulnerability-detection', 'security-health-analytics', 'virtual-machine-threat-detection', 'web-security-scanner'], default='none', help_str='Service name in Security Command Center').AddToParser(parser)