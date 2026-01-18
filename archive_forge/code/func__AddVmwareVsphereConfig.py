from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddVmwareVsphereConfig(vmware_node_config_group: parser_arguments.ArgumentInterceptor, release_track: base.ReleaseTrack=None, for_update: bool=False):
    """Adds a flag for VmwareVsphereConfig."""
    if for_update:
        return
    if release_track is None or release_track != base.ReleaseTrack.ALPHA:
        return
    vmware_vsphere_config_help_text = textwrap.dedent('    vSphere configurations for the node pool.\n\n    DATASTORE is the name of the vCenter datastore.\n\n    STORAGE_POLICY_NAME is the name of the vCenter storage policy.\n    ')
    vmware_node_config_group.add_argument('--vsphere-config', help=vmware_vsphere_config_help_text, hidden=True, type=arg_parsers.ArgDict(spec={'datastore': str, 'storage-policy-name': str}), metavar='datastore=DATASTORE,storage-policy-name=STORAGE_POLICY_NAME')