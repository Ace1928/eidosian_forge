from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _StopPreviousVersionIfApplies(old_default_version, api_client, wait_for_stop_version):
    """Stop the previous default version if applicable.

  Cases where a version will not be stopped:

  * If the previous default version is not serving, there is no need to stop it.
  * If the previous default version is an automatically scaled standard
    environment app, it cannot be stopped.

  Args:
    old_default_version: Version, The old default version to stop.
    api_client: appengine_api_client.AppengineApiClient to use to make requests.
    wait_for_stop_version: bool, indicating whether to wait for stop operation
    to finish.
  """
    version_object = old_default_version.version
    status_enum = api_client.messages.Version.ServingStatusValueValuesEnum
    if version_object.servingStatus != status_enum.SERVING:
        log.info('Previous default version [{0}] not serving, so not stopping it.'.format(old_default_version))
        return
    is_standard = not (version_object.vm or version_object.env == 'flex' or version_object.env == 'flexible')
    if is_standard and (not version_object.basicScaling) and (not version_object.manualScaling):
        log.info('Previous default version [{0}] is an automatically scaled standard environment app, so not stopping it.'.format(old_default_version))
        return
    log.status.Print('Stopping version [{0}].'.format(old_default_version))
    try:
        operations_util.CallAndCollectOpErrors(api_client.StopVersion, service_name=old_default_version.service, version_id=old_default_version.id, block=wait_for_stop_version)
    except operations_util.MiscOperationError as err:
        log.warning('Error stopping version [{0}]: {1}'.format(old_default_version, six.text_type(err)))
        log.warning('Version [{0}] is still running and you must stop or delete it yourself in order to turn it off. (If you do not, you may be charged.)'.format(old_default_version))
    else:
        if not wait_for_stop_version:
            log.status.Print('Sent request to stop version [{0}]. This operation may take some time to complete. If you would like to verify that it succeeded, run:\n  $ gcloud app versions describe -s {0.service} {0.id}\nuntil it shows that the version has stopped.'.format(old_default_version))