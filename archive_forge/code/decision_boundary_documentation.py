import numpy as np
from ...base import is_regressor
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_matplotlib_support
from ...utils._response import _get_response_values
from ...utils.validation import (
Plot decision boundary given an estimator.

        Read more in the :ref:`User Guide <visualizations>`.

        Parameters
        ----------
        estimator : object
            Trained estimator used to plot the decision boundary.

        X : {array-like, sparse matrix, dataframe} of shape (n_samples, 2)
            Input data that should be only 2-dimensional.

        grid_resolution : int, default=100
            Number of grid points to use for plotting decision boundary.
            Higher values will make the plot look nicer but be slower to
            render.

        eps : float, default=1.0
            Extends the minimum and maximum values of X for evaluating the
            response function.

        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.

        response_method : {'auto', 'predict_proba', 'decision_function',                 'predict'}, default='auto'
            Specifies whether to use :term:`predict_proba`,
            :term:`decision_function`, :term:`predict` as the target response.
            If set to 'auto', the response method is tried in the following order:
            :term:`decision_function`, :term:`predict_proba`, :term:`predict`.
            For multiclass problems, :term:`predict` is selected when
            `response_method="auto"`.

        class_of_interest : int, float, bool or str, default=None
            The class considered when plotting the decision. If None,
            `estimator.classes_[1]` is considered as the positive class
            for binary classifiers. For multiclass classifiers, passing
            an explicit value for `class_of_interest` is mandatory.

            .. versionadded:: 1.4

        xlabel : str, default=None
            The label used for the x-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ylabel : str, default=None
            The label used for the y-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keyword arguments to be passed to the
            `plot_method`.

        Returns
        -------
        display : :class:`~sklearn.inspection.DecisionBoundaryDisplay`
            Object that stores the result.

        See Also
        --------
        DecisionBoundaryDisplay : Decision boundary visualization.
        sklearn.metrics.ConfusionMatrixDisplay.from_estimator : Plot the
            confusion matrix given an estimator, the data, and the label.
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions : Plot the
            confusion matrix given the true and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.inspection import DecisionBoundaryDisplay
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]
        >>> classifier = LogisticRegression().fit(X, iris.target)
        >>> disp = DecisionBoundaryDisplay.from_estimator(
        ...     classifier, X, response_method="predict",
        ...     xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
        ...     alpha=0.5,
        ... )
        >>> disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
        <...>
        >>> plt.show()
        