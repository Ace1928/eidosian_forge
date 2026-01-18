import os
import pkgutil
import re
import sys
import warnings
from functools import partial
from inspect import isgenerator, signature
from itertools import chain, product
from pathlib import Path
import numpy as np
import pytest
import sklearn
from sklearn.cluster import (
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import (
from sklearn.neighbors import (
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import _IS_WASM, IS_PYPY, all_estimators
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
def _generate_search_cv_instances():
    for SearchCV, (Estimator, param_grid) in product([GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingGridSearchCV], [(Ridge, {'alpha': [0.1, 1.0]}), (LogisticRegression, {'C': [0.1, 1.0]})]):
        init_params = signature(SearchCV).parameters
        extra_params = {'min_resources': 'smallest'} if 'min_resources' in init_params else {}
        search_cv = SearchCV(Estimator(), param_grid, cv=2, **extra_params)
        set_random_state(search_cv)
        yield search_cv
    for SearchCV, (Estimator, param_grid) in product([GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV], [(Ridge, {'ridge__alpha': [0.1, 1.0]}), (LogisticRegression, {'logisticregression__C': [0.1, 1.0]})]):
        init_params = signature(SearchCV).parameters
        extra_params = {'min_resources': 'smallest'} if 'min_resources' in init_params else {}
        search_cv = SearchCV(make_pipeline(PCA(), Estimator()), param_grid, cv=2, **extra_params).set_params(error_score='raise')
        set_random_state(search_cv)
        yield search_cv