import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
def _load_and_remap_matrix_initializer(ckpt_path, old_tensor_name, new_row_vocab_size, new_col_vocab_size, old_row_vocab_size=-1, old_row_vocab_file=None, new_row_vocab_file=None, old_col_vocab_file=None, new_col_vocab_file=None, num_row_oov_buckets=0, num_col_oov_buckets=0, initializer=None, max_rows_in_memory=-1):
    """Returns a var initializer for loading and remapping a 2-D (matrix) tensor.

  The returned initializer loads a 2-D (matrix) `Tensor` with name
  `old_tensor_name` from the checkpoint at `ckpt_path`. It will reorder the
  rows/columns according to the specified vocab files and append additional
  out-of-vocabulary rows/columns according to the number of OOV buckets.

  The format of the file at the `{old,new}_{row,col}_vocab_file` path should be
  a text file, with each line containing a single entity within the vocabulary.
  Let the function `line_of(f, "x")` return the 0-indexed line number of the
  entity "x" in file f, and the function `entity_at(f, i)` return the entity at
  line i of file f. Then, row i of the new output matrix will be taken from row
  `line_of(old_row_vocab_file, entity_at(new_row_vocab_file, i))` of the old
  matrix. If any entity in `new_row_vocab_file` is not found in
  `old_row_vocab_file`, that row is considered a "missing" row, and its values
  will be initialized using the `initializer` arg. The same logic also applies
  for the columns.

  For example, assuming that:

  * `old_row_vocab_file` contains "mercury\\nvenus\\nmars"
  * `new_row_vocab_file` contains "venus\\njupiter\\nmercury"
  * `old_col_vocab_file` contains "good\\nbetter\\nbest"
  * `new_col_vocab_file` contains "good\\nbest\\nfantastic"
  * `initializer` returns the natural numbers `[1, 2, 3, 4, ...]`
  * `w(i, j)` represents the value from row i, column j of the old matrix

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1],
    [2,       3,       4],
    [w(0, 0), w(0, 2), 5]]`

  If we further specify that:

  * `num_row_oov_buckets` == 2
  * `num_col_oov_buckets` == 1

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1,  12],
    [2,       3,       4,  13],
    [w(0, 0), w(0, 2), 5,  14],
    [6,       7,       8,  15],
    [9,       10,      11, 16]]`

  If `{old,new}_row_vocab_file` are None, we assume that the old and new row
  vocab files are the same, and no row remapping is done. If
  `{old,new}_col_vocab_file` are None, we assume that the old and new column
  vocab files are the same, and no column remapping is done.

  The returned initializer only supports div-partitioning along the row axis. It
  does not support partitioning along the column axis (as this is not common in
  practice) or mod-partitioning.

  NOTE: When this is used to warm-start variables, client code should use
  `tf.lookup.index_table_from_tensor()` like
  contrib/layers/python/layers/feature_column.py does, as opposed to
  `tf.feature_to_id()` - in order to ensure the underlying lookup tables are the
  same.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_row_vocab_size: `int` specifying the number of entries in
      `new_row_vocab_file`. If no row remapping is needed (no row vocab
      provided), this should be equal to the number of rows to load from the old
      matrix (which can theoretically be smaller than the number of rows in the
      old matrix).
    new_col_vocab_size: `int` specifying the number of entries in
      `new_col_vocab_file`. If no column remapping is needed (no column vocab
      provided), this should be equal to the number of columns in the old
      matrix.
    old_row_vocab_size: The number of entries to consider in the old vocabulary.
      With the default value of -1, the entire old row vocabulary file will be
      used.  Otherwise, only the first `old_row_vocab_size` entries will be
      considered for remapping.Must be smaller than the length of
      `old_row_vocab_file`.  NOTE: we do not provide an equivalent
      `old_col_vocab_size` for classes.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    initializer: Initializer function to initialize missing values. Accepts a
      1-D tensor as the arg to specify the shape of the returned tensor. If
      `None`, defaults to using `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function that should be used to initialize a
    (potentially partitioned) `Variable` whose complete shape is
    `[new_row_vocab_size + num_row_oov_buckets, new_col_vocab_size +
    num_col_oov_buckets]`.

  Raises:
    TypeError: If `initializer` is specified but not callable.
  """
    if initializer is None:
        initializer = init_ops.zeros_initializer()
    if not callable(initializer):
        raise TypeError('initializer must be callable, instead of being {} of type {}.'.format(initializer, type(initializer)))

    def _initializer(shape, dtype=dtypes.float32, partition_info=None):
        """Variable initializer.

    Args:
      shape: Shape of `Tensor` to return. Should include OOV on both axes.
      dtype: Must be float32.
      partition_info: variable_scope._PartitionInfo.

    Returns:
      `Tensor` of shape `shape`.

    Raises:
      TypeError: If `dtype` is anything other than float32.
      ValueError: For shape mismatch upon invocation.
    """
        if dtype != dtypes.float32:
            raise TypeError('Currently, only float32 is supported. Received dtype: {}'.format(dtype))
        if len(shape) != 2:
            raise ValueError('Expected 2-dim shape, but received: {}'.format(shape))
        if shape[0] <= 0:
            raise ValueError('Expected 1st dim of shape to be > 0, but received shape: {}'.format(shape))
        if shape[1] != new_col_vocab_size + num_col_oov_buckets:
            raise ValueError('Expected 2nd dim of shape to be new_col_vocab_size ({}) + num_col_oov_buckets ({}) = {}, but received shape: {}'.format(new_col_vocab_size, num_col_oov_buckets, new_col_vocab_size + num_col_oov_buckets, shape))
        offset = 0
        if partition_info is not None:
            offset = partition_info.single_offset(shape)
        if offset + shape[0] > new_row_vocab_size + num_row_oov_buckets:
            raise ValueError('Trying to initialize {} additional rows after {} rows have already been initialized, which would exceed expected total row count of new_row_vocab_size ({}) + num_row_oov_buckets ({}) = {}.'.format(shape[0], offset, new_row_vocab_size, num_row_oov_buckets, new_row_vocab_size + num_row_oov_buckets))
        row_oov_buckets_to_use = min(shape[0], max(0, offset + shape[0] - new_row_vocab_size))
        num_rows_to_load = shape[0] - row_oov_buckets_to_use
        if offset > new_row_vocab_size:
            if shape[0] != row_oov_buckets_to_use:
                raise ValueError('Partitioned variable offset is greater than new vocab size and not operating on OOV-only partition.')
            return initializer(shape)
        return _load_and_remap_matrix(ckpt_path=ckpt_path, old_tensor_name=old_tensor_name, new_row_vocab_offset=offset, num_rows_to_load=num_rows_to_load, new_col_vocab_size=new_col_vocab_size, initializer=initializer, old_row_vocab_size=old_row_vocab_size, old_row_vocab_file=old_row_vocab_file, new_row_vocab_file=new_row_vocab_file, old_col_vocab_file=old_col_vocab_file, new_col_vocab_file=new_col_vocab_file, num_row_oov_buckets=row_oov_buckets_to_use, num_col_oov_buckets=num_col_oov_buckets, max_rows_in_memory=max_rows_in_memory)
    return _initializer