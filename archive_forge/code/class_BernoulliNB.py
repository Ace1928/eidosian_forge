import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import logsumexp
from .base import BaseEstimator, ClassifierMixin, _fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize
from .utils._param_validation import Interval
from .utils.extmath import safe_sparse_dot
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative
class BernoulliNB(_BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : The Complement Naive Bayes classifier
        described in Rennie et al. (2003).
    GaussianNB : Gaussian Naive Bayes (GaussianNB).
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """
    _parameter_constraints: dict = {**_BaseDiscreteNB._parameter_constraints, 'binarize': [None, Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, force_alpha=force_alpha)
        self.binarize = binarize

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = super()._check_X(X)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y, reset=True):
        X, y = super()._check_X_y(X, y, reset=reset)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return (X, y)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_features = self.feature_log_prob_.shape[1]
        n_features_X = X.shape[1]
        if n_features_X != n_features:
            raise ValueError('Expected input with %d features, got %d instead' % (n_features, n_features_X))
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)
        return jll