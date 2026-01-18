from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def AddInstanceSettingsArgs(parser):
    """Declare flag for instance settings."""
    parser.display_info.AddFormat(flags.GetInstanceListFormat())
    flags.AddActivationPolicy(parser, hidden=True)
    flags.AddActiveDirectoryDomain(parser, hidden=True)
    flags.AddAssignIp(parser, hidden=True)
    flags.AddAuthorizedNetworks(parser, hidden=True)
    flags.AddAvailabilityType(parser, hidden=True)
    flags.AddBackup(parser, hidden=True)
    flags.AddBackupStartTime(parser, hidden=True)
    flags.AddBackupLocation(parser, allow_empty=False, hidden=True)
    flags.AddCPU(parser, hidden=True)
    flags.AddInstanceCollation(parser, hidden=True)
    flags.AddDatabaseFlags(parser, hidden=True)
    flags.AddEnableBinLog(parser, hidden=True)
    flags.AddRetainedBackupsCount(parser, hidden=True)
    flags.AddRetainedTransactionLogDays(parser, hidden=True)
    flags.AddFailoverReplicaName(parser, hidden=True)
    flags.AddMaintenanceReleaseChannel(parser, hidden=True)
    flags.AddMaintenanceWindowDay(parser, hidden=True)
    flags.AddMaintenanceWindowHour(parser, hidden=True)
    flags.AddDenyMaintenancePeriodStartDate(parser, hidden=True)
    flags.AddDenyMaintenancePeriodEndDate(parser, hidden=True)
    flags.AddDenyMaintenancePeriodTime(parser, hidden=True)
    flags.AddInsightsConfigQueryInsightsEnabled(parser, hidden=True)
    flags.AddInsightsConfigQueryStringLength(parser, hidden=True)
    flags.AddInsightsConfigRecordApplicationTags(parser, hidden=True)
    flags.AddInsightsConfigRecordClientAddress(parser, hidden=True)
    flags.AddInsightsConfigQueryPlansPerMinute(parser, hidden=True)
    flags.AddMasterInstanceName(parser, hidden=True)
    flags.AddMemory(parser, hidden=True)
    flags.AddPasswordPolicyMinLength(parser, hidden=True)
    flags.AddPasswordPolicyComplexity(parser, hidden=True)
    flags.AddPasswordPolicyReuseInterval(parser, hidden=True)
    flags.AddPasswordPolicyDisallowUsernameSubstring(parser, hidden=True)
    flags.AddPasswordPolicyPasswordChangeInterval(parser, hidden=True)
    flags.AddPasswordPolicyEnablePasswordPolicy(parser, hidden=True)
    flags.AddReplicaType(parser, hidden=True)
    flags.AddReplication(parser, hidden=True)
    flags.AddRequireSsl(parser, hidden=True)
    flags.AddRootPassword(parser, hidden=True)
    flags.AddStorageAutoIncrease(parser, hidden=True)
    flags.AddStorageSize(parser, hidden=True)
    flags.AddStorageType(parser, hidden=True)
    flags.AddTier(parser, hidden=True)
    flags.AddEdition(parser, hidden=True)
    kms_flag_overrides = {'kms-key': '--disk-encryption-key', 'kms-keyring': '--disk-encryption-key-keyring', 'kms-location': '--disk-encryption-key-location', 'kms-project': '--disk-encryption-key-project'}
    kms_resource_args.AddKmsKeyResourceArg(parser, 'instance', flag_overrides=kms_flag_overrides, hidden=True)
    flags.AddEnablePointInTimeRecovery(parser, hidden=True)
    flags.AddNetwork(parser, hidden=True)
    flags.AddSqlServerAudit(parser, hidden=True)
    flags.AddDeletionProtection(parser, hidden=True)
    flags.AddSqlServerTimeZone(parser, hidden=True)
    flags.AddConnectorEnforcement(parser, hidden=True)
    flags.AddTimeout(parser, _INSTANCE_CREATION_TIMEOUT_SECONDS, hidden=True)
    flags.AddEnableGooglePrivatePath(parser, show_negated_in_help=False, hidden=True)
    flags.AddThreadsPerCore(parser, hidden=True)
    flags.AddCascadableReplica(parser, hidden=True)
    flags.AddEnableDataCache(parser, show_negated_in_help=False, hidden=True)
    flags.AddRecreateReplicasOnPrimaryCrash(parser, hidden=True)
    psc_setup_group = parser.add_group(hidden=True)
    flags.AddEnablePrivateServiceConnect(psc_setup_group, hidden=True)
    flags.AddAllowedPscProjects(psc_setup_group, hidden=True)
    flags.AddSslMode(parser, hidden=True)
    flags.AddEnableGoogleMLIntegration(parser, hidden=True)
    flags.AddLocationGroup(parser, hidden=True, specify_default_region=False)
    flags.AddDatabaseVersion(parser, restrict_choices=False, hidden=True, support_default_version=False)