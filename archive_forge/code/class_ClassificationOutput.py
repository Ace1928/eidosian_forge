import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class ClassificationOutput(ExportOutput):
    """Represents the output of a classification head.

  Either classes or scores or both must be set.

  The classes `Tensor` must provide string labels, not integer class IDs.

  If only classes is set, it is interpreted as providing top-k results in
  descending order.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
  """

    def __init__(self, scores=None, classes=None):
        """Constructor for `ClassificationOutput`.

    Args:
      scores: A float `Tensor` giving scores (sometimes but not always
          interpretable as probabilities) for each class.  May be `None`, but
          only if `classes` is set.  Interpretation varies-- see class doc.
      classes: A string `Tensor` giving predicted class labels.  May be `None`,
          but only if `scores` is set.  Interpretation varies-- see class doc.

    Raises:
      ValueError: if neither classes nor scores is set, or one of them is not a
          `Tensor` with the correct dtype.
    """
        if scores is not None and (not (isinstance(scores, tensor.Tensor) and scores.dtype.is_floating)):
            raise ValueError('Classification scores must be a float32 Tensor; got {}'.format(scores))
        if classes is not None and (not (isinstance(classes, tensor.Tensor) and dtypes.as_dtype(classes.dtype) == dtypes.string)):
            raise ValueError('Classification classes must be a string Tensor; got {}'.format(classes))
        if scores is None and classes is None:
            raise ValueError('At least one of scores and classes must be set.')
        self._scores = scores
        self._classes = classes

    @property
    def scores(self):
        return self._scores

    @property
    def classes(self):
        return self._classes

    def as_signature_def(self, receiver_tensors):
        if len(receiver_tensors) != 1:
            raise ValueError('Classification input must be a single string Tensor; got {}'.format(receiver_tensors))
        (_, examples), = receiver_tensors.items()
        if dtypes.as_dtype(examples.dtype) != dtypes.string:
            raise ValueError('Classification input must be a single string Tensor; got {}'.format(receiver_tensors))
        return signature_def_utils.classification_signature_def(examples, self.classes, self.scores)