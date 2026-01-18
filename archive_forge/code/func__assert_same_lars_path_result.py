import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def _assert_same_lars_path_result(output1, output2):
    assert len(output1) == len(output2)
    for o1, o2 in zip(output1, output2):
        assert_allclose(o1, o2)