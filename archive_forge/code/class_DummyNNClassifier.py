from functools import partial
import numpy as np
import pytest
from scipy import spatial
from skimage.future import fit_segmenter, predict_segmenter, TrainableSegmenter
from skimage.feature import multiscale_basic_features
class DummyNNClassifier:

    def fit(self, X, labels):
        self.X = X
        self.labels = labels
        self.tree = spatial.cKDTree(self.X)

    def predict(self, X):
        if X.shape[1] != self.X.shape[1]:
            raise ValueError(f'Expected {self.X.shape[1]} features but got {X.shape[1]}.')
        nearest_neighbors = self.tree.query(X)[1]
        return self.labels[nearest_neighbors]