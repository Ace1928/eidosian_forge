from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationSliceSlice(_messages.Message):
    """Definition of a slice.

  Fields:
    dimension: Output only. The dimension of the slice. Well-known dimensions
      are: * `annotationSpec`: This slice is on the test data that has either
      ground truth or prediction with AnnotationSpec.display_name equals to
      value. * `slice`: This slice is a user customized slice defined by its
      SliceSpec.
    sliceSpec: Output only. Specification for how the data was sliced.
    value: Output only. The value of the dimension in this slice.
  """
    dimension = _messages.StringField(1)
    sliceSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpec', 2)
    value = _messages.StringField(3)