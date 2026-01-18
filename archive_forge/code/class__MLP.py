from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
class _MLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super(_MLP, self).__init__()
        self.hidden_size = round((input_size + num_classes) / 2)
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        hidden = self.fc1(x)
        r1 = self.relu(hidden)
        out = self.fc2(r1)
        return out