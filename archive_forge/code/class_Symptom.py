from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Symptom(_messages.Message):
    """A Symptom instance.

  Enums:
    SymptomTypeValueValuesEnum: Type of the Symptom.

  Fields:
    createTime: Timestamp when the Symptom is created.
    details: Detailed information of the current Symptom.
    symptomType: Type of the Symptom.
    workerId: A string used to uniquely distinguish a worker within a TPU
      node.
  """

    class SymptomTypeValueValuesEnum(_messages.Enum):
        """Type of the Symptom.

    Values:
      SYMPTOM_TYPE_UNSPECIFIED: Unspecified symptom.
      LOW_MEMORY: TPU VM memory is low.
      OUT_OF_MEMORY: TPU runtime is out of memory.
      EXECUTE_TIMED_OUT: TPU runtime execution has timed out.
      MESH_BUILD_FAIL: TPU runtime fails to construct a mesh that recognizes
        each TPU device's neighbors.
      HBM_OUT_OF_MEMORY: TPU HBM is out of memory.
      PROJECT_ABUSE: Abusive behaviors have been identified on the current
        project.
    """
        SYMPTOM_TYPE_UNSPECIFIED = 0
        LOW_MEMORY = 1
        OUT_OF_MEMORY = 2
        EXECUTE_TIMED_OUT = 3
        MESH_BUILD_FAIL = 4
        HBM_OUT_OF_MEMORY = 5
        PROJECT_ABUSE = 6
    createTime = _messages.StringField(1)
    details = _messages.StringField(2)
    symptomType = _messages.EnumField('SymptomTypeValueValuesEnum', 3)
    workerId = _messages.StringField(4)