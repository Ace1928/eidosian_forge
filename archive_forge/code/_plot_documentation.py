import warnings
import numpy as np
from ..utils import check_matplotlib_support
from ..utils._plotting import _interval_max_min_ratio, _validate_score_name
from ._validation import learning_curve, validation_curve
Create a validation curve display from an estimator.

        Read more in the :ref:`User Guide <visualizations>` for general
        information about the visualization API and :ref:`detailed
        documentation <validation_curve>` regarding the validation curve
        visualization.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        param_name : str
            Name of the parameter that will be varied.

        param_range : array-like of shape (n_values,)
            The values of the parameter that will be evaluated.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`GroupKFold`).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For int/None inputs, if the estimator is a classifier and `y` is
            either binary or multiclass,
            :class:`~sklearn.model_selection.StratifiedKFold` is used. In all
            other cases, :class:`~sklearn.model_selection.KFold` is used. These
            splitters are instantiated with `shuffle=False` so the splits will
            be the same across calls.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        scoring : str or callable, default=None
            A string (see :ref:`scoring_parameter`) or
            a scorer callable object / function with signature
            `scorer(estimator, X, y)` (see :ref:`scoring`).

        n_jobs : int, default=None
            Number of jobs to run in parallel. Training the estimator and
            computing the score are parallelized over the different training
            and test sets. `None` means 1 unless in a
            :obj:`joblib.parallel_backend` context. `-1` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.

        pre_dispatch : int or str, default='all'
            Number of predispatched jobs for parallel execution (default is
            all). The option can reduce the allocated memory. The str can
            be an expression like '2*n_jobs'.

        verbose : int, default=0
            Controls the verbosity: the higher, the more messages.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric value
            is given, FitFailedWarning is raised.

        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.validation_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If `None`, no representation of the standard deviation
            is displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.ValidationCurveDisplay`
            Object that stores computed values.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import ValidationCurveDisplay
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(n_samples=1_000, random_state=0)
        >>> logistic_regression = LogisticRegression()
        >>> param_name, param_range = "C", np.logspace(-8, 3, 10)
        >>> ValidationCurveDisplay.from_estimator(
        ...     logistic_regression, X, y, param_name=param_name,
        ...     param_range=param_range,
        ... )
        <...>
        >>> plt.show()
        