from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddRemoteLoginFlags(parser, for_adc=False):
    if for_adc:
        auth_command = 'gcloud auth application-default login'
        auth_target = 'client libraries'
    else:
        auth_command = 'gcloud auth login'
        auth_target = 'gcloud CLI'
    AddNoBrowserArgGroup(parser, auth_target, auth_command)
    AddNoLaunchBrowserFlag(parser, auth_target, auth_command)