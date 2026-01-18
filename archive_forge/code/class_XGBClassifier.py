import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
@xgboost_model_doc('Implementation of the scikit-learn API for XGBoost classification.', ['model', 'objective'], extra_parameters='\n    n_estimators : Optional[int]\n        Number of boosting rounds.\n')
class XGBClassifier(XGBModel, XGBClassifierBase):

    @_deprecate_positional_args
    def __init__(self, *, objective: SklObjective='binary:logistic', **kwargs: Any) -> None:
        super().__init__(objective=objective, **kwargs)

    @_deprecate_positional_args
    def fit(self, X: ArrayLike, y: ArrayLike, *, sample_weight: Optional[ArrayLike]=None, base_margin: Optional[ArrayLike]=None, eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]]=None, eval_metric: Optional[Union[str, Sequence[str], Metric]]=None, early_stopping_rounds: Optional[int]=None, verbose: Optional[Union[bool, int]]=True, xgb_model: Optional[Union[Booster, str, XGBModel]]=None, sample_weight_eval_set: Optional[Sequence[ArrayLike]]=None, base_margin_eval_set: Optional[Sequence[ArrayLike]]=None, feature_weights: Optional[ArrayLike]=None, callbacks: Optional[Sequence[TrainingCallback]]=None) -> 'XGBClassifier':
        with config_context(verbosity=self.verbosity):
            evals_result: TrainingCallback.EvalsLog = {}
            if _is_cudf_df(y) or _is_cudf_ser(y):
                import cupy as cp
                classes = cp.unique(y.values)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            elif _is_cupy_array(y):
                import cupy as cp
                classes = cp.unique(y)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            else:
                classes = np.unique(np.asarray(y))
                self.n_classes_ = len(classes)
                expected_classes = self.classes_
            if classes.shape != expected_classes.shape or not (classes == expected_classes).all():
                raise ValueError(f'Invalid classes inferred from unique values of `y`.  Expected: {expected_classes}, got {classes}')
            params = self.get_xgb_params()
            if callable(self.objective):
                obj: Optional[Objective] = _objective_decorator(self.objective)
                params['objective'] = 'binary:logistic'
            else:
                obj = None
            if self.n_classes_ > 2:
                if params.get('objective', None) != 'multi:softmax':
                    params['objective'] = 'multi:softprob'
                params['num_class'] = self.n_classes_
            model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(xgb_model, eval_metric, params, early_stopping_rounds, callbacks)
            train_dmatrix, evals = _wrap_evaluation_matrices(missing=self.missing, X=X, y=y, group=None, qid=None, sample_weight=sample_weight, base_margin=base_margin, feature_weights=feature_weights, eval_set=eval_set, sample_weight_eval_set=sample_weight_eval_set, base_margin_eval_set=base_margin_eval_set, eval_group=None, eval_qid=None, create_dmatrix=self._create_dmatrix, enable_categorical=self.enable_categorical, feature_types=self.feature_types)
            self._Booster = train(params, train_dmatrix, self.get_num_boosting_rounds(), evals=evals, early_stopping_rounds=early_stopping_rounds, evals_result=evals_result, obj=obj, custom_metric=metric, verbose_eval=verbose, xgb_model=model, callbacks=callbacks)
            if not callable(self.objective):
                self.objective = params['objective']
            self._set_evaluation_result(evals_result)
            return self
    assert XGBModel.fit.__doc__ is not None
    fit.__doc__ = XGBModel.fit.__doc__.replace('Fit gradient boosting model', 'Fit gradient boosting classifier', 1)

    def predict(self, X: ArrayLike, output_margin: bool=False, validate_features: bool=True, base_margin: Optional[ArrayLike]=None, iteration_range: Optional[Tuple[int, int]]=None) -> ArrayLike:
        with config_context(verbosity=self.verbosity):
            class_probs = super().predict(X=X, output_margin=output_margin, validate_features=validate_features, base_margin=base_margin, iteration_range=iteration_range)
            if output_margin:
                return class_probs
            if len(class_probs.shape) > 1 and self.n_classes_ != 2:
                column_indexes: np.ndarray = np.argmax(class_probs, axis=1)
            elif len(class_probs.shape) > 1 and class_probs.shape[1] != 1:
                column_indexes = np.zeros(class_probs.shape)
                column_indexes[class_probs > 0.5] = 1
            elif self.objective == 'multi:softmax':
                return class_probs.astype(np.int32)
            else:
                column_indexes = np.repeat(0, class_probs.shape[0])
                column_indexes[class_probs > 0.5] = 1
            return column_indexes

    def predict_proba(self, X: ArrayLike, validate_features: bool=True, base_margin: Optional[ArrayLike]=None, iteration_range: Optional[Tuple[int, int]]=None) -> np.ndarray:
        """Predict the probability of each `X` example being of a given class. If the
        model is trained with early stopping, then :py:attr:`best_iteration` is used
        automatically. The estimator uses `inplace_predict` by default and falls back to
        using :py:class:`DMatrix` if devices between the data and the estimator don't
        match.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X :
            Feature matrix. See :ref:`py-data` for a list of supported types.
        validate_features :
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin :
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

        Returns
        -------
        prediction :
            a numpy array of shape array-like of shape (n_samples, n_classes) with the
            probability of each data example being of a given class.

        """
        if self.objective == 'multi:softmax':
            raw_predt = super().predict(X=X, validate_features=validate_features, base_margin=base_margin, iteration_range=iteration_range, output_margin=True)
            class_prob = softmax(raw_predt, axis=1)
            return class_prob
        class_probs = super().predict(X=X, validate_features=validate_features, base_margin=base_margin, iteration_range=iteration_range)
        return _cls_predict_proba(self.n_classes_, class_probs, np.vstack)

    @property
    def classes_(self) -> np.ndarray:
        return np.arange(self.n_classes_)