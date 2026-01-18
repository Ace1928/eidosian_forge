from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SdkHarnessContainerImage(_messages.Message):
    """Defines an SDK harness container for executing Dataflow pipelines.

  Fields:
    capabilities: The set of capabilities enumerated in the above Environment
      proto. See also [beam_runner_api.proto](https://github.com/apache/beam/b
      lob/master/model/pipeline/src/main/proto/org/apache/beam/model/pipeline/
      v1/beam_runner_api.proto)
    containerImage: A docker container image that resides in Google Container
      Registry.
    environmentId: Environment ID for the Beam runner API proto Environment
      that corresponds to the current SDK Harness.
    useSingleCorePerContainer: If true, recommends the Dataflow service to use
      only one core per SDK container instance with this image. If false (or
      unset) recommends using more than one core per SDK container instance
      with this image for efficiency. Note that Dataflow service may choose to
      override this property if needed.
  """
    capabilities = _messages.StringField(1, repeated=True)
    containerImage = _messages.StringField(2)
    environmentId = _messages.StringField(3)
    useSingleCorePerContainer = _messages.BooleanField(4)