from functools import partial
import numpy as np
from . import _catboost
class BuiltinMetric(object):

    @staticmethod
    def params_with_defaults():
        """
        For each valid metric parameter, returns its default value and if this parameter is mandatory.
        Implemented in child classes.

        Returns
        ----------
        valid_params: dict: param_name -> {'default_value': default value or None, 'is_mandatory': bool}
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def __str__(self):
        """
        Gets the representation of the metric object with overridden parameters.
        Implemented in child classes.

        Returns
        ----------
        metric_string: str representing the metric object.
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def set_hints(self, **hints):
        """
        Sets hints for the metric. Hints are not validated.
        Implemented in child classes.

        Returns
        ----------
        self: for chained calls.
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def eval(self, label, approx, weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs=None, thread_count=-1):
        """
        Evaluate the metric with raw approxes and labels.

        Parameters
        ----------
        label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            Object labels.

        approx : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            Object approxes.

        weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Object weights.

        group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Object group ids.

        group_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Group weights.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs : list or numpy.ndarray or pandas.DataFrame or string or pathlib.Path
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of winner object in the training set. The second element of the pair is
            the index of loser object in the training set.
            If string or pathlib.Path, giving the path to the file with pairs description.

        thread_count : int, optional (default=-1)
            Number of threads to work with.
            If -1, then the number of threads is set to the number of CPU cores.

        Returns
        -------
        metric results : list with metric values.
        """
        if len(label) > 0 and (not isinstance(label[0], _ARRAY_TYPES)):
            label = [label]
        if len(approx) == 0:
            approx = [[]]
        if not isinstance(approx[0], _ARRAY_TYPES):
            approx = [approx]
        return _catboost._eval_metric_util(label, approx, str(self), weight, group_id, group_weight, subgroup_id, pairs, thread_count)

    def is_max_optimal(self):
        """
        Returns
        ----------
        bool : True if metric is maximizable, False otherwise
        """
        return _catboost.is_maximizable_metric(str(self))

    def is_min_optimal(self):
        """
        Returns
        ----------
        bool :  True if metric is minimizable, False otherwise
        """
        return _catboost.is_minimizable_metric(str(self))