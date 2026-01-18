import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
def _load_embedding_initializer(ckpt_path, embedding_tensor_name, new_vocab_size, embedding_dim, old_vocab_file, new_vocab_file, old_vocab_size=-1, num_oov_buckets=0, initializer=None, max_rows_in_memory=-1):
    """Returns a variable initializer for loading pre-trained embeddings.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  embedding weights and remapping according to the provided vocab files. See
  docs for `load_and_remap_matrix_initializer()` for more details.

  NOTE: Only for use with div-partitioned variables / vocabularies.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    embedding_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_vocab_size: Number of entries in the new vocab.
    embedding_dim: `int` specifying the dimension of the embedding vectors from
      the checkpoint. Must match the number of columns in the old embedding
      matrix.
    old_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old vocabulary file.
    new_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the new vocabulary file.
    old_vocab_size: The number of entries to consider in the old vocabulary.
      With the default value of -1, the entire old row vocabulary file will be
      used.  Otherwise, only the first `old_vocab_size` entries will be
      considered for remapping.Must be smaller than the length of
      `old_row_vocab_file`.
    num_oov_buckets: `int` specifying the number of out-of-vocabulary
      buckets to use. Must be >= 0.
    initializer: Initializer function that accepts a 1-D tensor as the arg to
      specify the shape of the returned tensor. If `None`, defaults to using
      `truncated_normal_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function.
  """
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_dim))
    return _load_and_remap_matrix_initializer(ckpt_path=ckpt_path, old_tensor_name=embedding_tensor_name, new_row_vocab_size=new_vocab_size, new_col_vocab_size=embedding_dim, old_row_vocab_size=old_vocab_size, old_row_vocab_file=old_vocab_file, new_row_vocab_file=new_vocab_file, old_col_vocab_file=None, new_col_vocab_file=None, num_row_oov_buckets=num_oov_buckets, num_col_oov_buckets=0, initializer=initializer, max_rows_in_memory=max_rows_in_memory)