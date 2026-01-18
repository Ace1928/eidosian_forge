from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1PodStatus(_messages.Message):
    """A GoogleCloudApigeeV1PodStatus object.

  Fields:
    appVersion: Version of the application running in the pod.
    deploymentStatus: Status of the deployment. Valid values include: -
      `deployed`: Successful. - `error` : Failed. - `pending` : Pod has not
      yet reported on the deployment.
    deploymentStatusTime: Time the deployment status was reported in
      milliseconds since epoch.
    deploymentTime: Time the proxy was deployed in milliseconds since epoch.
    podName: Name of the pod which is reporting the status.
    podStatus: Overall status of the pod (not this specific deployment). Valid
      values include: - `active`: Up to date. - `stale` : Recently out of
      date. Pods that have not reported status in a long time are excluded
      from the output.
    podStatusTime: Time the pod status was reported in milliseconds since
      epoch.
    statusCode: Code associated with the deployment status.
    statusCodeDetails: Human-readable message associated with the status code.
  """
    appVersion = _messages.StringField(1)
    deploymentStatus = _messages.StringField(2)
    deploymentStatusTime = _messages.IntegerField(3)
    deploymentTime = _messages.IntegerField(4)
    podName = _messages.StringField(5)
    podStatus = _messages.StringField(6)
    podStatusTime = _messages.IntegerField(7)
    statusCode = _messages.StringField(8)
    statusCodeDetails = _messages.StringField(9)