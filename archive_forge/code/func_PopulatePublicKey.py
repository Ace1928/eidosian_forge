from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.compute import base_classes as compute_base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def PopulatePublicKey(api_client, service_id, version_id, instance_id, public_key, release_track):
    """Enable debug mode on and send SSH keys to a flex instance.

  Common method for SSH-like commands, does the following:
  - Makes sure that the service/version/instance specified exists and is of the
    right type (Flexible).
  - If not already done, prompts and enables debug on the instance.
  - Populates the public key onto the instance.

  Args:
    api_client: An appengine_api_client.AppEngineApiClient.
    service_id: str, The service ID.
    version_id: str, The version ID.
    instance_id: str, The instance ID.
    public_key: ssh.Keys.PublicKey, Public key to send.
    release_track: calliope.base.ReleaseTrack, The current release track.

  Raises:
    InvalidInstanceTypeError: The instance is not supported for SSH.
    MissingVersionError: The version specified does not exist.
    MissingInstanceError: The instance specified does not exist.
    UnattendedPromptError: Not running in a tty.
    OperationCancelledError: User cancelled the operation.

  Returns:
    ConnectionDetails, the details to use for SSH/SCP for the SSH
    connection.
  """
    try:
        version = api_client.GetVersionResource(service=service_id, version=version_id)
    except apitools_exceptions.HttpNotFoundError:
        raise command_exceptions.MissingVersionError('{}/{}'.format(service_id, version_id))
    version = version_util.Version.FromVersionResource(version, None)
    if version.environment is not env.FLEX:
        if version.environment is env.MANAGED_VMS:
            environment = 'Managed VMs'
            msg = 'Use `gcloud compute ssh` for Managed VMs instances.'
        else:
            environment = 'Standard'
            msg = None
        raise command_exceptions.InvalidInstanceTypeError(environment, msg)
    res = resources.REGISTRY.Parse(instance_id, params={'appsId': properties.VALUES.core.project.GetOrFail, 'versionsId': version_id, 'instancesId': instance_id, 'servicesId': service_id}, collection='appengine.apps.services.versions.instances')
    rel_name = res.RelativeName()
    try:
        instance = api_client.GetInstanceResource(res)
    except apitools_exceptions.HttpNotFoundError:
        raise command_exceptions.MissingInstanceError(rel_name)
    if not instance.vmDebugEnabled:
        log.warning(_ENABLE_DEBUG_WARNING)
        console_io.PromptContinue(cancel_on_no=True, throw_if_unattended=True)
    user = ssh.GetDefaultSshUsername()
    project = _GetComputeProject(release_track)
    oslogin_state = ssh.GetOsloginState(None, project, user, public_key.ToEntry(), None, release_track, messages=compute_base_classes.ComputeApiHolder(release_track).client.messages)
    user = oslogin_state.user
    instance_ip_mode_enum = api_client.messages.Network.InstanceIpModeValueValuesEnum
    host = instance.id if version.version.network.instanceIpMode is instance_ip_mode_enum.INTERNAL else instance.vmIp
    remote = ssh.Remote(host=host, user=user)
    if not oslogin_state.oslogin_enabled:
        ssh_key = '{user}:{key} {user}'.format(user=user, key=public_key.ToEntry())
        log.status.Print('Sending public key to instance [{}].'.format(rel_name))
        api_client.DebugInstance(res, ssh_key)
    options = {'IdentitiesOnly': 'yes', 'UserKnownHostsFile': ssh.KnownHosts.DEFAULT_PATH, 'CheckHostIP': 'no', 'HostKeyAlias': _HOST_KEY_ALIAS.format(project=api_client.project, instance_id=instance_id)}
    return ConnectionDetails(remote, options)