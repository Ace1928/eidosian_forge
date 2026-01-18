import math
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.dataset_utils import is_torch_tensor
from keras.src.utils.nest import lists_to_tuples
Slice inputs into a Dataset of batches.

            Given a Dataset of batch indices and the unsliced inputs,
            this step slices the inputs in a parallelized fashion
            and produces a dataset of input batches.

            Args:
                indices_dataset: A Dataset of batched indices.
                inputs: A python data structure that contains the inputs,
                    targets, and possibly sample weights.

            Returns:
                A Dataset of input batches matching the batch indices.
            