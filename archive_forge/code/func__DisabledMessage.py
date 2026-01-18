from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
def _DisabledMessage(instance, release_track):
    command_args = ['compute', 'project_info', 'add-metadata', '--metadata=enable-osconfig=true']
    project_command = utils.GetCommandString(command_args, release_track)
    instance_args = ['compute', 'instances', 'add-metadata', instance.name, '--metadata=enable-osconfig=true']
    instance_command = utils.GetCommandString(instance_args, release_track)
    return 'No\nOS Config agent is not enabled for this VM instance. To enable for all VMs in this project, run:\n\n' + project_command + '\n\nTo enable for this VM, run:\n\n' + instance_command