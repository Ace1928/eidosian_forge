import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _warn_or_raise_about_fit_failures(results, error_score):
    fit_errors = [result['fit_error'] for result in results if result['fit_error'] is not None]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = '-' * 80 + '\n'
        fit_errors_summary = '\n'.join((f'{delimiter}{n} fits failed with the following error:\n{error}' for error, n in fit_errors_counter.items()))
        if num_failed_fits == num_fits:
            all_fits_failed_message = f"\nAll the {num_fits} fits failed.\nIt is very likely that your model is misconfigured.\nYou can try to debug the error by setting error_score='raise'.\n\nBelow are more details about the failures:\n{fit_errors_summary}"
            raise ValueError(all_fits_failed_message)
        else:
            some_fits_failed_message = f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\nThe score on these train-test partitions for these parameters will be set to {error_score}.\nIf these failures are not expected, you can try to debug them by setting error_score='raise'.\n\nBelow are more details about the failures:\n{fit_errors_summary}"
            warnings.warn(some_fits_failed_message, FitFailedWarning)