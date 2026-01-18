from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesPatchRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesPatchRequest object.

  Fields:
    name: Output only. The resource name of the runtime. Format:
      `projects/{project}/locations/{location}/runtimes/{runtimeId}`
    requestId: Idempotent request UUID.
    runtime: A Runtime resource to be passed as the request body.
    updateMask: Required. Specifies the path, relative to `Runtime`, of the
      field to update. For example, to change the software configuration
      kernels, the `update_mask` parameter would be specified as
      `software_config.kernels`, and the `PATCH` request body would specify
      the new value, as follows: { "software_config":{ "kernels": [{
      'repository': 'gcr.io/deeplearning-platform-release/pytorch-gpu', 'tag':
      'latest' }], } } Currently, only the following fields can be updated: -
      `software_config.kernels` - `software_config.post_startup_script` -
      `software_config.custom_gpu_driver_path` -
      `software_config.idle_shutdown` -
      `software_config.idle_shutdown_timeout` -
      `software_config.disable_terminal` - `labels`
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    runtime = _messages.MessageField('Runtime', 3)
    updateMask = _messages.StringField(4)