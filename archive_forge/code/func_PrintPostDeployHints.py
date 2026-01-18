from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import scheduler
from googlecloudsdk.api_lib import tasks
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import deploy_app_command_util
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.datastore import index_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.tasks import app_deploy_migration_util
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import create_util
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.command_lib.app import source_files_util
from googlecloudsdk.command_lib.app import staging
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def PrintPostDeployHints(new_versions, updated_configs):
    """Print hints for user at the end of a deployment."""
    if yaml_parsing.ConfigYamlInfo.CRON in updated_configs:
        log.status.Print('\nCron jobs have been updated.')
        if yaml_parsing.ConfigYamlInfo.QUEUE not in updated_configs:
            log.status.Print('\nVisit the Cloud Platform Console Task Queues page to view your queues and cron jobs.')
            log.status.Print(_TASK_CONSOLE_LINK.format(properties.VALUES.core.project.Get()))
    if yaml_parsing.ConfigYamlInfo.DISPATCH in updated_configs:
        log.status.Print('\nCustom routings have been updated.')
    if yaml_parsing.ConfigYamlInfo.QUEUE in updated_configs:
        log.status.Print('\nTask queues have been updated.')
        log.status.Print('\nVisit the Cloud Platform Console Task Queues page to view your queues and cron jobs.')
    if yaml_parsing.ConfigYamlInfo.INDEX in updated_configs:
        log.status.Print('\nIndexes are being rebuilt. This may take a moment.')
    if not new_versions:
        return
    elif len(new_versions) > 1:
        service_hint = ' -s <service>'
    elif new_versions[0].service == 'default':
        service_hint = ''
    else:
        service = new_versions[0].service
        service_hint = ' -s {svc}'.format(svc=service)
    proj_conf = named_configs.ActivePropertiesFile.Load().Get('core', 'project')
    project = properties.VALUES.core.project.Get()
    if proj_conf != project:
        project_hint = ' --project=' + project
    else:
        project_hint = ''
    log.status.Print('\nYou can stream logs from the command line by running:\n  $ gcloud app logs tail' + (service_hint or ' -s default'))
    log.status.Print('\nTo view your application in the web browser run:\n  $ gcloud app browse' + service_hint + project_hint)