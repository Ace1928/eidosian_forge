from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
A substitute for `Dataset.interleave()` on a fixed list of datasets.