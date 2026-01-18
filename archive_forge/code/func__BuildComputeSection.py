from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _BuildComputeSection(instances, private_key_file, known_hosts_file):
    """Returns a string representing the Compute section that should be added."""
    temp_buf = []
    for instance in instances:
        external_ip_address = ssh_utils.GetExternalIPAddress(instance, no_raise=True)
        host_key_alias = 'compute.{0}'.format(instance.id)
        if external_ip_address:
            temp_buf.extend(textwrap.dedent('          Host {alias}\n              HostName {external_ip_address}\n              IdentityFile {private_key_file}\n              UserKnownHostsFile={known_hosts_file}\n              HostKeyAlias={host_key_alias}\n              IdentitiesOnly=yes\n              CheckHostIP=no\n\n          '.format(alias=_CreateAlias(instance), external_ip_address=external_ip_address, private_key_file=private_key_file, known_hosts_file=known_hosts_file, host_key_alias=host_key_alias)))
    if temp_buf:
        buf = io.StringIO()
        buf.write(_HEADER)
        for i in temp_buf:
            buf.write(i)
        buf.write(_END_MARKER)
        buf.write('\n')
        return buf.getvalue()
    else:
        return ''