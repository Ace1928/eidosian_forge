import multiprocessing
import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
class _LoadDataset(dataset_ops.DatasetSource):
    """A dataset that loads previously saved dataset."""

    def __init__(self, path, element_spec, compression, reader_func):
        self._path = path
        self._element_spec = element_spec
        self._compression = compression
        self._reader_func = structured_function.StructuredFunctionWrapper(reader_func, 'load()', input_structure=dataset_ops.DatasetSpec(dataset_ops.DatasetSpec(self._element_spec)))
        variant_tensor = ged_ops.load_dataset(path, reader_func_other_args=self._reader_func.function.captured_inputs, compression=compression, reader_func=self._reader_func.function, **self._flat_structure)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec