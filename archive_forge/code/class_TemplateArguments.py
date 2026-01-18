from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class TemplateArguments:
    """Wrapper class for template arguments."""
    project_id = None
    region_id = None
    gcs_location = None
    job_name = None
    zone = None
    max_workers = None
    num_workers = None
    network = None
    subnetwork = None
    worker_machine_type = None
    staging_location = None
    temp_location = None
    kms_key_name = None
    disable_public_ips = None
    parameters = None
    service_account_email = None
    worker_region = None
    worker_zone = None
    enable_streaming_engine = None
    additional_experiments = None
    additional_user_labels = None
    streaming_update = None
    transform_name_mappings = None
    flexrs_goal = None

    def __init__(self, project_id=None, region_id=None, job_name=None, gcs_location=None, zone=None, max_workers=None, num_workers=None, network=None, subnetwork=None, worker_machine_type=None, staging_location=None, temp_location=None, kms_key_name=None, disable_public_ips=None, parameters=None, service_account_email=None, worker_region=None, worker_zone=None, enable_streaming_engine=None, additional_experiments=None, additional_user_labels=None, streaming_update=None, transform_name_mappings=None, flexrs_goal=None):
        self.project_id = project_id
        self.region_id = region_id
        self.job_name = job_name
        self.gcs_location = gcs_location
        self.zone = zone
        self.max_workers = max_workers
        self.num_workers = num_workers
        self.network = network
        self.subnetwork = subnetwork
        self.worker_machine_type = worker_machine_type
        self.staging_location = staging_location
        self.temp_location = temp_location
        self.kms_key_name = kms_key_name
        self.disable_public_ips = disable_public_ips
        self.parameters = parameters
        self.service_account_email = service_account_email
        self.worker_region = worker_region
        self.worker_zone = worker_zone
        self.enable_streaming_engine = enable_streaming_engine
        self.additional_experiments = additional_experiments
        self.additional_user_labels = additional_user_labels
        self.streaming_update = streaming_update
        self.transform_name_mappings = transform_name_mappings
        self.flexrs_goal = flexrs_goal