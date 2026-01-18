from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
class VMBootTroubleshooter(ssh_troubleshooter.SshTroubleshooter):
    """Check VM boot and kernel panic issues.

  Attributes:
    project: The project object.
    zone: str, the zone name.
    instance: The instance object.
  """

    def __init__(self, project, zone, instance):
        self.project = project
        self.zone = zone
        self.instance = instance
        self.compute_client = apis.GetClientInstance(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_message = apis.GetMessagesModule(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.issues = {}

    def check_prerequisite(self):
        return

    def cleanup_resources(self):
        return

    def troubleshoot(self):
        log.status.Print('---- Checking VM boot status ----')
        sc_log = ssh_troubleshooter_utils.GetSerialConsoleLog(self.compute_client, self.compute_message, self.instance.name, self.project.name, self.zone)
        if ssh_troubleshooter_utils.SearchPatternErrorInLog(VM_BOOT_PATTERNS, sc_log):
            self.issues['boot_issue'] = VM_BOOT_MESSAGE
        if ssh_troubleshooter_utils.SearchPatternErrorInLog(KERNEL_PANIC_PATTERNS, sc_log):
            self.issues['kernel_panic'] = KERNEL_PANIC_MESSAGE
        log.status.Print('VM boot: {0} issue(s) found.\n'.format(len(self.issues)))
        for message in self.issues.values():
            log.status.Print(message)