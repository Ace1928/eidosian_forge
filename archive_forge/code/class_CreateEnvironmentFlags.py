from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
class CreateEnvironmentFlags:
    """Container holding environment creation flag values.

  Attributes:
    node_count: int or None, the number of VMs to create for the environment
    environment_size: str or None, one of small, medium and large.
    labels: dict(str->str), a dict of user-provided resource labels to apply to
      the environment and its downstream resources
    location: str or None, the Compute Engine zone in which to create the
      environment specified as relative resource name.
    machine_type: str or None, the Compute Engine machine type of the VMs to
      create specified as relative resource name.
    network: str or None, the Compute Engine network to which to connect the
      environment specified as relative resource name.
    subnetwork: str or None, the Compute Engine subnetwork to which to connect
      the environment specified as relative resource name.
    network_attachment: str or None, the Compute Engine network attachment that
      is used as PSC Network entry point.
    env_variables: dict(str->str), a dict of user-provided environment variables
      to provide to the Airflow scheduler, worker, and webserver processes.
    airflow_config_overrides: dict(str->str), a dict of user-provided Airflow
      configuration overrides.
    service_account: str or None, the user-provided service account
    oauth_scopes: [str], the user-provided OAuth scopes
    tags: [str], the user-provided networking tags
    disk_size_gb: int, the disk size of node VMs, in GB
    python_version: str or None, major python version to use within created
      environment.
    image_version: str or None, the desired image for created environment in the
      format of 'composer-(version)-airflow-(version)'
    airflow_executor_type: str or None, the airflow executor type to run task
      instances.
    use_ip_aliases: bool or None, create env cluster nodes using alias IPs.
    cluster_secondary_range_name: str or None, the name of secondary range to
      allocate IP addresses to pods in GKE cluster.
    services_secondary_range_name: str or None, the name of the secondary range
      to allocate IP addresses to services in GKE cluster.
    cluster_ipv4_cidr_block: str or None, the IP address range to allocate IP
      adresses to pods in GKE cluster.
    services_ipv4_cidr_block: str or None, the IP address range to allocate IP
      addresses to services in GKE cluster.
    max_pods_per_node: int or None, the maximum number of pods that can be
      assigned to a GKE cluster node.
    enable_ip_masq_agent: bool or None, when enabled, the GKE IP Masq Agent is
      deployed to the cluster.
    private_environment: bool or None, create env cluster nodes with no public
      IP addresses.
    private_endpoint: bool or None, managed env cluster using the private IP
      address of the master API endpoint.
    master_ipv4_cidr: IPv4 CIDR range to use for the cluster master network.
    privately_used_public_ips: bool or None, when enabled, GKE pod and services
      can use IPs from public (non-RFC1918) ranges.
    web_server_ipv4_cidr: IPv4 CIDR range to use for Web Server network.
    cloud_sql_ipv4_cidr: IPv4 CIDR range to use for Cloud SQL network.
    composer_network_ipv4_cidr: IPv4 CIDR range to use for Composer network.
    connection_subnetwork: str or None, the Compute Engine subnetwork from which
      to reserve the IP address for internal connections, specified as relative
      resource name.
    connection_type: str or None, mode of internal connectivity within the Cloud
      Composer environment. Can be VPC_PEERING or PRIVATE_SERVICE_CONNECT.
    web_server_access_control: [{string: string}], List of IP ranges with
      descriptions to allow access to the web server.
    cloud_sql_machine_type: str or None, Cloud SQL machine type used by the
      Airflow database.
    cloud_sql_preferred_zone: str or None, Cloud SQL db preferred zone. Can be
      specified only in Composer 2.0.0.
    web_server_machine_type: str or None, machine type used by the Airflow web
      server
    kms_key: str or None, the user-provided customer-managed encryption key
      resource name
    scheduler_cpu: float or None, CPU allocated to Airflow scheduler. Can be
      specified only in Composer 2.0.0.
    worker_cpu: float or None, CPU allocated to each Airflow worker. Can be
      specified only in Composer 2.0.0.
    web_server_cpu: float or None, CPU allocated to Airflow web server. Can be
      specified only in Composer 2.0.0.
    scheduler_memory_gb: float or None, memory allocated to Airflow scheduler.
      Can be specified only in Composer 2.0.0.
    worker_memory_gb: float or None, memory allocated to each Airflow worker.
      Can be specified only in Composer 2.0.0.
    web_server_memory_gb: float or None, memory allocated to Airflow web server.
      Can be specified only in Composer 2.0.0.
    scheduler_storage_gb: float or None, storage allocated to Airflow scheduler.
      Can be specified only in Composer 2.0.0.
    worker_storage_gb: float or None, storage allocated to each Airflow worker.
      Can be specified only in Composer 2.0.0.
    web_server_storage_gb: float or None, storage allocated to Airflow web
      server. Can be specified only in Composer 2.0.0.
    min_workers: int or None, minimum number of workers in the Environment. Can
      be specified only in Composer 2.0.0.
    max_workers: int or None, maximum number of workers in the Environment. Can
      be specified only in Composer 2.0.0.
    scheduler_count: int or None, number of schedulers in the Environment.
    maintenance_window_start: Datetime or None, the starting time of the
      maintenance window
    maintenance_window_end: Datetime or None, the ending time of the maintenance
      window
    maintenance_window_recurrence: str or None, the recurrence of the
      maintenance window
    enable_master_authorized_networks: bool or None, whether master authorized
      networks should be enabled
    master_authorized_networks: list(str), master authorized networks
    airflow_database_retention_days: Optional[int], the number of retention days
      for airflow database data retention mechanism. Infinite retention will be
      applied in case `0` or no integer is provided.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.
    enable_triggerer: bool or None, enable triggerer in the Environment. Can be
      specified only in Airflow 2.2.x and greater
    triggerer_cpu: float or None, CPU allocated to Airflow triggerer. Can be
      specified only in Airflow 2.2.x and greater
    triggerer_count: int or None, number of Airflow triggerers. Can be specified
      only in Airflow 2.2.x and greater
    triggerer_memory_gb: float or None, memory allocated to Airflow triggerer.
      Can be specified only in Airflow 2.2.x and greater
    enable_scheduled_snapshot_creation: bool or None, whether the automatic
      snapshot creation should be enabled
    snapshot_creation_schedule: str or None, cron expression that specifies when
      snapshots will be created
    snapshot_location: str or None, a Cloud Storage location used to store
      automatically created snapshots
    snapshot_schedule_timezone: str or None, time zone that sets the context to
      interpret snapshot_creation_schedule
    enable_cloud_data_lineage_integration: bool or None, whether Cloud Data
      Lineage integration should be enabled
    disable_cloud_data_lineage_integration: bool or None, whether Cloud Data
      Lineage integration should be disabled
    enable_high_resilience: bool or None, whether high resilience should be
      enabled
    enable_logs_in_cloud_logging_only: bool or None, whether logs in cloud
      logging only should be enabled
    disable_logs_in_cloud_logging_only: bool or None, whether logs in cloud
      logging only should be disabled
    support_web_server_plugins: bool or None, whether to enable/disable the
      support for web server plugins
    dag_processor_cpu: float or None, CPU allocated to Airflow dag processor.
      Can be specified only in Composer 3.
    dag_processor_count: int or None, number of Airflow dag processors. Can be
      specified only in Composer 3.
    dag_processor_memory_gb: float or None, memory allocated to Airflow dag
      processor. Can be specified only in Composer 3.
    dag_processor_storage_gb: float or None, storage allocated to Airflow dag
      processor. Can be specified only in Composer 3.
    composer_internal_ipv4_cidr_block: str or None. The IP range in CIDR
      notation to use internally by Cloud Composer. Can be specified only in
      Composer 3.
    enable_private_builds_only: bool or None, whether to enable the support for
      private only builds.
    disable_private_builds_only: bool or None, whether to disable the support
      for private only builds.
    storage_bucket: str or None. An existing Cloud Storage bucket to be used by
      the environment.
  """

    def __init__(self, node_count=None, environment_size=None, labels=None, location=None, machine_type=None, network=None, subnetwork=None, network_attachment=None, env_variables=None, airflow_config_overrides=None, service_account=None, oauth_scopes=None, tags=None, disk_size_gb=None, python_version=None, image_version=None, airflow_executor_type=None, use_ip_aliases=None, cluster_secondary_range_name=None, services_secondary_range_name=None, cluster_ipv4_cidr_block=None, services_ipv4_cidr_block=None, max_pods_per_node=None, enable_ip_masq_agent=None, private_environment=None, private_endpoint=None, master_ipv4_cidr=None, privately_used_public_ips=None, web_server_ipv4_cidr=None, cloud_sql_ipv4_cidr=None, composer_network_ipv4_cidr=None, connection_subnetwork=None, connection_type=None, web_server_access_control=None, cloud_sql_machine_type=None, web_server_machine_type=None, kms_key=None, scheduler_cpu=None, worker_cpu=None, web_server_cpu=None, scheduler_memory_gb=None, worker_memory_gb=None, web_server_memory_gb=None, scheduler_storage_gb=None, worker_storage_gb=None, web_server_storage_gb=None, min_workers=None, max_workers=None, scheduler_count=None, maintenance_window_start=None, maintenance_window_end=None, maintenance_window_recurrence=None, enable_master_authorized_networks=None, master_authorized_networks=None, airflow_database_retention_days=None, release_track=base.ReleaseTrack.GA, enable_triggerer=None, triggerer_cpu=None, triggerer_count=None, triggerer_memory_gb=None, enable_scheduled_snapshot_creation=None, snapshot_creation_schedule=None, snapshot_location=None, snapshot_schedule_timezone=None, enable_cloud_data_lineage_integration=None, disable_cloud_data_lineage_integration=None, enable_high_resilience=None, enable_logs_in_cloud_logging_only=None, disable_logs_in_cloud_logging_only=None, cloud_sql_preferred_zone=None, support_web_server_plugins=None, dag_processor_cpu=None, dag_processor_count=None, dag_processor_memory_gb=None, dag_processor_storage_gb=None, composer_internal_ipv4_cidr_block=None, enable_private_builds_only=None, disable_private_builds_only=None, storage_bucket=None):
        self.node_count = node_count
        self.environment_size = environment_size
        self.labels = labels
        self.location = location
        self.machine_type = machine_type
        self.network = network
        self.subnetwork = subnetwork
        self.network_attachment = network_attachment
        self.env_variables = env_variables
        self.airflow_config_overrides = airflow_config_overrides
        self.service_account = service_account
        self.oauth_scopes = oauth_scopes
        self.tags = tags
        self.disk_size_gb = disk_size_gb
        self.python_version = python_version
        self.image_version = image_version
        self.airflow_executor_type = airflow_executor_type
        self.use_ip_aliases = use_ip_aliases
        self.cluster_secondary_range_name = cluster_secondary_range_name
        self.services_secondary_range_name = services_secondary_range_name
        self.cluster_ipv4_cidr_block = cluster_ipv4_cidr_block
        self.services_ipv4_cidr_block = services_ipv4_cidr_block
        self.max_pods_per_node = max_pods_per_node
        self.enable_ip_masq_agent = enable_ip_masq_agent
        self.private_environment = private_environment
        self.private_endpoint = private_endpoint
        self.master_ipv4_cidr = master_ipv4_cidr
        self.privately_used_public_ips = privately_used_public_ips
        self.web_server_ipv4_cidr = web_server_ipv4_cidr
        self.cloud_sql_ipv4_cidr = cloud_sql_ipv4_cidr
        self.composer_network_ipv4_cidr = composer_network_ipv4_cidr
        self.connection_subnetwork = connection_subnetwork
        self.connection_type = connection_type
        self.web_server_access_control = web_server_access_control
        self.cloud_sql_machine_type = cloud_sql_machine_type
        self.web_server_machine_type = web_server_machine_type
        self.kms_key = kms_key
        self.scheduler_cpu = scheduler_cpu
        self.worker_cpu = worker_cpu
        self.web_server_cpu = web_server_cpu
        self.scheduler_memory_gb = scheduler_memory_gb
        self.worker_memory_gb = worker_memory_gb
        self.web_server_memory_gb = web_server_memory_gb
        self.scheduler_storage_gb = scheduler_storage_gb
        self.worker_storage_gb = worker_storage_gb
        self.web_server_storage_gb = web_server_storage_gb
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scheduler_count = scheduler_count
        self.enable_triggerer = enable_triggerer
        self.triggerer_cpu = triggerer_cpu
        self.triggerer_count = triggerer_count
        self.triggerer_memory_gb = triggerer_memory_gb
        self.maintenance_window_start = maintenance_window_start
        self.maintenance_window_end = maintenance_window_end
        self.maintenance_window_recurrence = maintenance_window_recurrence
        self.enable_master_authorized_networks = enable_master_authorized_networks
        self.master_authorized_networks = master_authorized_networks
        self.airflow_database_retention_days = airflow_database_retention_days
        self.release_track = release_track
        self.enable_scheduled_snapshot_creation = enable_scheduled_snapshot_creation
        self.snapshot_creation_schedule = snapshot_creation_schedule
        self.snapshot_location = snapshot_location
        self.snapshot_schedule_timezone = snapshot_schedule_timezone
        self.enable_cloud_data_lineage_integration = enable_cloud_data_lineage_integration
        self.disable_cloud_data_lineage_integration = disable_cloud_data_lineage_integration
        self.enable_high_resilience = enable_high_resilience
        self.enable_logs_in_cloud_logging_only = enable_logs_in_cloud_logging_only
        self.disable_logs_in_cloud_logging_only = disable_logs_in_cloud_logging_only
        self.cloud_sql_preferred_zone = cloud_sql_preferred_zone
        self.support_web_server_plugins = support_web_server_plugins
        self.dag_processor_cpu = dag_processor_cpu
        self.dag_processor_storage_gb = dag_processor_storage_gb
        self.dag_processor_memory_gb = dag_processor_memory_gb
        self.dag_processor_count = dag_processor_count
        self.composer_internal_ipv4_cidr_block = composer_internal_ipv4_cidr_block
        self.enable_private_builds_only = enable_private_builds_only
        self.disable_private_builds_only = disable_private_builds_only
        self.storage_bucket = storage_bucket