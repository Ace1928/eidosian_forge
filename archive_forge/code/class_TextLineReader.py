from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['TextLineReader'])
class TextLineReader(ReaderBase):
    """A Reader that outputs the lines of a file delimited by newlines.

  Newlines are stripped from the output.
  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TextLineDataset`.')
    def __init__(self, skip_header_lines=None, name=None):
        """Create a TextLineReader.

    Args:
      skip_header_lines: An optional int. Defaults to 0.  Number of lines
        to skip from the beginning of every file.
      name: A name for the operation (optional).
    """
        rr = gen_io_ops.text_line_reader_v2(skip_header_lines=skip_header_lines, name=name)
        super(TextLineReader, self).__init__(rr)