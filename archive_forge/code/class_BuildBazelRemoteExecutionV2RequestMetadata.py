from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2RequestMetadata(_messages.Message):
    """An optional Metadata to attach to any RPC request to tell the server
  about an external context of the request. The server may use this for
  logging or other purposes. To use it, the client attaches the header to the
  call using the canonical proto serialization: * name:
  `build.bazel.remote.execution.v2.requestmetadata-bin` * contents: the base64
  encoded binary `RequestMetadata` message. Note: the gRPC library serializes
  binary headers encoded in base64 by default
  (https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests).
  Therefore, if the gRPC library is used to pass/retrieve this metadata, the
  user may ignore the base64 encoding and assume it is simply serialized as a
  binary message.

  Fields:
    actionId: An identifier that ties multiple requests to the same action.
      For example, multiple requests to the CAS, Action Cache, and Execution
      API are used in order to compile foo.cc.
    actionMnemonic: A brief description of the kind of action, for example,
      CppCompile or GoLink. There is no standard agreed set of values for
      this, and they are expected to vary between different client tools.
    configurationId: An identifier for the configuration in which the target
      was built, e.g. for differentiating building host tools or different
      target platforms. There is no expectation that this value will have any
      particular structure, or equality across invocations, though some client
      tools may offer these guarantees.
    correlatedInvocationsId: An identifier to tie multiple tool invocations
      together. For example, runs of foo_test, bar_test and baz_test on a
      post-submit of a given patch.
    targetId: An identifier for the target which produced this action. No
      guarantees are made around how many actions may relate to a single
      target.
    toolDetails: The details for the tool invoking the requests.
    toolInvocationId: An identifier that ties multiple actions together to a
      final result. For example, multiple actions are required to build and
      run foo_test.
  """
    actionId = _messages.StringField(1)
    actionMnemonic = _messages.StringField(2)
    configurationId = _messages.StringField(3)
    correlatedInvocationsId = _messages.StringField(4)
    targetId = _messages.StringField(5)
    toolDetails = _messages.MessageField('BuildBazelRemoteExecutionV2ToolDetails', 6)
    toolInvocationId = _messages.StringField(7)