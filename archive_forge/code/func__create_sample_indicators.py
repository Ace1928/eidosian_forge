import numpy as np
import pandas as pd
import graphtools
from . import utils
from . import filter
from graphtools.estimator import GraphEstimator, attribute
from functools import partial
def _create_sample_indicators(self, sample_labels):
    """
        Helper function to take an array-like of non-numerics and produce a collection
        of sample indicator vectors.
        """
    self.sample_labels_ = sample_labels
    self.samples = np.unique(sample_labels)
    try:
        labels = sample_labels.values
    except AttributeError:
        labels = self.sample_labels_
    if len(labels.shape) > 1:
        if labels.shape[1] == 1:
            labels = labels.reshape(-1)
        else:
            raise ValueError('sample_labels must be a single column. Gotshape={}'.format(labels.shape))
    if self.samples.shape[0] == 2:
        df = pd.DataFrame([labels == self.samples[0], labels == self.samples[1]], columns=self._labels_index).astype(int)
        df.index = self.samples
        self.sample_indicators = df.T
    else:
        import sklearn
        self._LB = sklearn.preprocessing.LabelBinarizer()
        sample_indicators = self._LB.fit_transform(self.sample_labels_)
        self.sample_indicators = pd.DataFrame(sample_indicators, columns=self._LB.classes_)
    return self.sample_indicators