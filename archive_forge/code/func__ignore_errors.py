from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
def _ignore_errors(input_dataset, log_warning=False, name=None):
    return _IgnoreErrorsDataset(input_dataset, log_warning, name)