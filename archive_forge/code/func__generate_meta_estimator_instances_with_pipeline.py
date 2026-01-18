import functools
from inspect import signature
import numpy as np
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import (
from sklearn.utils.validation import check_is_fitted
def _generate_meta_estimator_instances_with_pipeline():
    """Generate instances of meta-estimators fed with a pipeline

    Are considered meta-estimators all estimators accepting one of "estimator",
    "base_estimator" or "estimators".
    """
    for _, Estimator in sorted(all_estimators()):
        sig = set(signature(Estimator).parameters)
        if 'estimator' in sig or 'base_estimator' in sig or 'regressor' in sig:
            if is_regressor(Estimator):
                estimator = make_pipeline(TfidfVectorizer(), Ridge())
                param_grid = {'ridge__alpha': [0.1, 1.0]}
            else:
                estimator = make_pipeline(TfidfVectorizer(), LogisticRegression())
                param_grid = {'logisticregression__C': [0.1, 1.0]}
            if 'param_grid' in sig or 'param_distributions' in sig:
                extra_params = {'n_iter': 2} if 'n_iter' in sig else {}
                yield Estimator(estimator, param_grid, **extra_params)
            else:
                yield Estimator(estimator)
        elif 'transformer_list' in sig:
            transformer_list = [('trans1', make_pipeline(TfidfVectorizer(), MaxAbsScaler())), ('trans2', make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False)))]
            yield Estimator(transformer_list)
        elif 'estimators' in sig:
            if is_regressor(Estimator):
                estimator = [('est1', make_pipeline(TfidfVectorizer(), Ridge(alpha=0.1))), ('est2', make_pipeline(TfidfVectorizer(), Ridge(alpha=1)))]
            else:
                estimator = [('est1', make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1))), ('est2', make_pipeline(TfidfVectorizer(), LogisticRegression(C=1)))]
            yield Estimator(estimator)
        else:
            continue