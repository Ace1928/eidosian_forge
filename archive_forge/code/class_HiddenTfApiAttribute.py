import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
class HiddenTfApiAttribute(property):
    """Hides a class attribute from the public API.

  Attributes in public classes can be hidden from the API by having an '_' in
  front of the name (e.g. ClassName._variables). This doesn't work when
  attributes or methods are inherited from a parent class. To hide inherited
  attributes, set their values to be `deprecation.hide_attribute_from_api`.
  For example, this is used in V2 Estimator to hide the deprecated
  export_savedmodel method:
    class EstimatorV2(Estimator):
       export_savedmodel = deprecation.hide_attribute_from_api('...')
  """

    def __init__(self, deprecation_message):

        def raise_error(unused_self):
            raise AttributeError(deprecation_message)
        super(HiddenTfApiAttribute, self).__init__(raise_error)