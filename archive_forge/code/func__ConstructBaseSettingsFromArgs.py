from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@classmethod
def _ConstructBaseSettingsFromArgs(cls, sql_messages, args, instance=None, release_track=DEFAULT_RELEASE_TRACK):
    """Constructs instance settings from the command line arguments.

    Args:
      sql_messages: module, The messages module that should be used.
      args: argparse.Namespace, The arguments that this command was invoked
        with.
      instance: sql_messages.DatabaseInstance, The original instance, for
        settings that depend on the previous state.
      release_track: base.ReleaseTrack, the release track that this was run
        under.

    Returns:
      A settings object representing the instance settings.

    Raises:
      ToolException: An error other than http error occurred while executing the
          command.
    """
    if 'authorized_gae_apps' not in args:
        args.authorized_gae_apps = None
    if 'follow_gae_app' not in args:
        args.follow_gae_app = None
    if 'pricing_plan' not in args:
        args.pricing_plan = 'PER_USE'
    settings = sql_messages.Settings(kind='sql#settings', tier=reducers.MachineType(instance, args.tier, args.memory, args.cpu), pricingPlan=_ParsePricingPlan(sql_messages, args.pricing_plan), replicationType=_ParseReplicationType(sql_messages, args.replication), activationPolicy=_ParseActivationPolicy(sql_messages, args.activation_policy))
    if args.authorized_gae_apps:
        settings.authorizedGaeApplications = args.authorized_gae_apps
    if any([args.assign_ip is not None, args.require_ssl is not None, args.authorized_networks]):
        settings.ipConfiguration = sql_messages.IpConfiguration()
        if args.assign_ip is not None:
            cls.SetIpConfigurationEnabled(settings, args.assign_ip)
        if args.authorized_networks:
            cls.SetAuthorizedNetworks(settings, args.authorized_networks, sql_messages.AclEntry)
        if args.require_ssl is not None:
            settings.ipConfiguration.requireSsl = args.require_ssl
    if any([args.follow_gae_app, _GetZone(args), _GetSecondaryZone(args)]):
        settings.locationPreference = sql_messages.LocationPreference(kind='sql#locationPreference', followGaeApplication=args.follow_gae_app, zone=_GetZone(args), secondaryZone=_GetSecondaryZone(args))
    if args.storage_size:
        settings.dataDiskSizeGb = int(args.storage_size / constants.BYTES_TO_GB)
    if args.storage_auto_increase is not None:
        settings.storageAutoResize = args.storage_auto_increase
    if args.IsSpecified('availability_type'):
        settings.availabilityType = _ParseAvailabilityType(sql_messages, args.availability_type)
    if args.IsSpecified('network'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        settings.ipConfiguration.privateNetwork = reducers.PrivateNetworkUrl(args.network)
    if args.IsKnownAndSpecified('enable_google_private_path'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        settings.ipConfiguration.enablePrivatePathForGoogleCloudServices = args.enable_google_private_path
    if args.IsKnownAndSpecified('enable_private_service_connect'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        if not settings.ipConfiguration.pscConfig:
            settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
        settings.ipConfiguration.pscConfig.pscEnabled = args.enable_private_service_connect
    if args.IsSpecified('allowed_psc_projects'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        if not settings.ipConfiguration.pscConfig:
            settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
        settings.ipConfiguration.pscConfig.allowedConsumerProjects = args.allowed_psc_projects
    if args.IsKnownAndSpecified('clear_allowed_psc_projects'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        if not settings.ipConfiguration.pscConfig:
            settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
        settings.ipConfiguration.pscConfig.allowedConsumerProjects = []
    if args.deletion_protection is not None:
        settings.deletionProtectionEnabled = args.deletion_protection
    if args.IsSpecified('connector_enforcement'):
        settings.connectorEnforcement = _ParseConnectorEnforcement(sql_messages, args.connector_enforcement)
    if args.recreate_replicas_on_primary_crash is not None:
        settings.recreateReplicasOnPrimaryCrash = args.recreate_replicas_on_primary_crash
    if args.IsSpecified('edition'):
        settings.edition = _ParseEdition(sql_messages, args.edition)
    if args.IsKnownAndSpecified('enable_data_cache'):
        if not settings.dataCacheConfig:
            settings.dataCacheConfig = sql_messages.DataCacheConfig()
        settings.dataCacheConfig.dataCacheEnabled = args.enable_data_cache
    if args.IsSpecified('ssl_mode'):
        if not settings.ipConfiguration:
            settings.ipConfiguration = sql_messages.IpConfiguration()
        settings.ipConfiguration.sslMode = _ParseSslMode(sql_messages, args.ssl_mode)
    if args.enable_google_ml_integration is not None:
        settings.enableGoogleMlIntegration = args.enable_google_ml_integration
    if IsBetaOrNewer(release_track):
        if args.IsSpecified('storage_auto_increase_limit'):
            if instance and instance.settings.storageAutoResize or args.storage_auto_increase:
                settings.storageAutoResizeLimit = args.storage_auto_increase_limit or 0
            else:
                raise exceptions.RequiredArgumentException('--storage-auto-increase', 'To set the storage capacity limit using [--storage-auto-increase-limit], [--storage-auto-increase] must be enabled.')
        if args.replication_lag_max_seconds_for_recreate is not None:
            settings.replicationLagMaxSeconds = args.replication_lag_max_seconds_for_recreate
    return settings