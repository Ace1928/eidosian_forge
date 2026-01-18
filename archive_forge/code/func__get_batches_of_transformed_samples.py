import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def _get_batches_of_transformed_samples(self, index_array):
    batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=self.dtype)
    for i, j in enumerate(index_array):
        x = self.x[j]
        params = self.image_data_generator.get_random_transform(x.shape)
        x = self.image_data_generator.apply_transform(x.astype(self.dtype), params)
        x = self.image_data_generator.standardize(x)
        batch_x[i] = x
    if self.save_to_dir:
        for i, j in enumerate(index_array):
            img = image_utils.array_to_img(batch_x[i], self.data_format, scale=True)
            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix, index=j, hash=np.random.randint(10000.0), format=self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))
    batch_x_miscs = [xx[index_array] for xx in self.x_misc]
    output = (batch_x if not batch_x_miscs else [batch_x] + batch_x_miscs,)
    if self.y is None:
        return output[0]
    output += (self.y[index_array],)
    if self.sample_weight is not None:
        output += (self.sample_weight[index_array],)
    return output