from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import config
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import promote_util
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
def _CheckSupportedVersion(self, release_ref, skaffold_version):
    config_client = config.ConfigClient()
    try:
        c = config_client.GetConfig(release_ref.AsDict()['projectsId'], release_ref.AsDict()['locationsId'])
    except apitools_exceptions.HttpForbiddenError:
        return
    version_obj = None
    for v in c.supportedVersions:
        if v.version == skaffold_version:
            version_obj = v
            break
    if not version_obj:
        return
    try:
        maintenance_dt = times.ParseDateTime(version_obj.maintenanceModeTime)
    except (times.DateTimeSyntaxError, times.DateTimeValueError):
        maintenance_dt = None
    try:
        support_expiration_dt = times.ParseDateTime(version_obj.supportExpirationTime)
    except (times.DateTimeSyntaxError, times.DateTimeValueError):
        support_expiration_dt = None
    if maintenance_dt and maintenance_dt - times.Now() <= datetime.timedelta(days=28):
        log.status.Print("WARNING: This release's Skaffold version will be in maintenance mode beginning on {date}. After that you won't be able to create releases using this version of Skaffold.\nhttps://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy".format(date=maintenance_dt.strftime('%Y-%m-%d')))
    if support_expiration_dt and times.Now() > support_expiration_dt:
        raise core_exceptions.Error("The Skaffold version you've chosen is no longer supported.\nhttps://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")
    if maintenance_dt and times.Now() > maintenance_dt:
        raise core_exceptions.Error("You can't create a new release using a Skaffold version that is in maintenance mode.\nhttps://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")