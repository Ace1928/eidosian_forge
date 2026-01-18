from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeTraceConfig(_messages.Message):
    """NEXT ID: 8 RuntimeTraceConfig defines the configurations for distributed
  trace in an environment.

  Enums:
    ExporterValueValuesEnum: Exporter that is used to view the distributed
      trace captured using OpenCensus. An exporter sends traces to any backend
      that is capable of consuming them. Recorded spans can be exported by
      registered exporters.

  Fields:
    endpoint: Endpoint of the exporter.
    exporter: Exporter that is used to view the distributed trace captured
      using OpenCensus. An exporter sends traces to any backend that is
      capable of consuming them. Recorded spans can be exported by registered
      exporters.
    name: Name of the trace config in the following format:
      `organizations/{org}/environment/{env}/traceConfig`
    overrides: List of trace configuration overrides for spicific API proxies.
    revisionCreateTime: The timestamp that the revision was created or
      updated.
    revisionId: Revision number which can be used by the runtime to detect if
      the trace config has changed between two versions.
    samplingConfig: Trace configuration for all API proxies in an environment.
  """

    class ExporterValueValuesEnum(_messages.Enum):
        """Exporter that is used to view the distributed trace captured using
    OpenCensus. An exporter sends traces to any backend that is capable of
    consuming them. Recorded spans can be exported by registered exporters.

    Values:
      EXPORTER_UNSPECIFIED: Exporter unspecified
      JAEGER: Jaeger exporter
      CLOUD_TRACE: Cloudtrace exporter
    """
        EXPORTER_UNSPECIFIED = 0
        JAEGER = 1
        CLOUD_TRACE = 2
    endpoint = _messages.StringField(1)
    exporter = _messages.EnumField('ExporterValueValuesEnum', 2)
    name = _messages.StringField(3)
    overrides = _messages.MessageField('GoogleCloudApigeeV1RuntimeTraceConfigOverride', 4, repeated=True)
    revisionCreateTime = _messages.StringField(5)
    revisionId = _messages.StringField(6)
    samplingConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeTraceSamplingConfig', 7)