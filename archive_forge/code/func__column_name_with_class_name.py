import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _column_name_with_class_name(fc):
    """Returns a unique name for the feature column used during deduping.

  Without this two FeatureColumns that have the same name and where
  one wraps the other, such as an IndicatorColumn wrapping a
  SequenceCategoricalColumn, will fail to deserialize because they will have the
  same name in columns_by_name, causing the wrong column to be returned.

  Args:
    fc: A FeatureColumn.

  Returns:
    A unique name as a string.
  """
    return fc.__class__.__name__ + ':' + fc.name