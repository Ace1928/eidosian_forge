from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddNoBrowserArgGroup(parser, auth_target, auth_command):
    group = parser.add_mutually_exclusive_group()
    AddNoBrowserFlag(group, auth_target, auth_command)
    AddRemoteBootstrapFlag(group)