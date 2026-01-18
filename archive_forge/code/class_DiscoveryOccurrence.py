from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoveryOccurrence(_messages.Message):
    """Provides information about the analysis status of a discovered resource.

  Enums:
    AnalysisStatusValueValuesEnum: The status of discovery for the resource.
    ContinuousAnalysisValueValuesEnum: Whether the resource is continuously
      analyzed.

  Fields:
    analysisCompleted: A AnalysisCompleted attribute.
    analysisError: Indicates any errors encountered during analysis of a
      resource. There could be 0 or more of these errors.
    analysisStatus: The status of discovery for the resource.
    analysisStatusError: When an error is encountered this will contain a
      LocalizedMessage under details to show to the user. The LocalizedMessage
      is output only and populated by the API.
    archiveTime: Output only. The time occurrences related to this discovery
      occurrence were archived.
    continuousAnalysis: Whether the resource is continuously analyzed.
    cpe: The CPE of the resource being scanned.
    lastScanTime: The last time this resource was scanned.
    sbomStatus: The status of an SBOM generation.
  """

    class AnalysisStatusValueValuesEnum(_messages.Enum):
        """The status of discovery for the resource.

    Values:
      ANALYSIS_STATUS_UNSPECIFIED: Unknown.
      PENDING: Resource is known but no action has been taken yet.
      SCANNING: Resource is being analyzed.
      FINISHED_SUCCESS: Analysis has finished successfully.
      COMPLETE: Analysis has completed.
      FINISHED_FAILED: Analysis has finished unsuccessfully, the analysis
        itself is in a bad state.
      FINISHED_UNSUPPORTED: The resource is known not to be supported.
    """
        ANALYSIS_STATUS_UNSPECIFIED = 0
        PENDING = 1
        SCANNING = 2
        FINISHED_SUCCESS = 3
        COMPLETE = 4
        FINISHED_FAILED = 5
        FINISHED_UNSUPPORTED = 6

    class ContinuousAnalysisValueValuesEnum(_messages.Enum):
        """Whether the resource is continuously analyzed.

    Values:
      CONTINUOUS_ANALYSIS_UNSPECIFIED: Unknown.
      ACTIVE: The resource is continuously analyzed.
      INACTIVE: The resource is ignored for continuous analysis.
    """
        CONTINUOUS_ANALYSIS_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
    analysisCompleted = _messages.MessageField('AnalysisCompleted', 1)
    analysisError = _messages.MessageField('Status', 2, repeated=True)
    analysisStatus = _messages.EnumField('AnalysisStatusValueValuesEnum', 3)
    analysisStatusError = _messages.MessageField('Status', 4)
    archiveTime = _messages.StringField(5)
    continuousAnalysis = _messages.EnumField('ContinuousAnalysisValueValuesEnum', 6)
    cpe = _messages.StringField(7)
    lastScanTime = _messages.StringField(8)
    sbomStatus = _messages.MessageField('SBOMStatus', 9)