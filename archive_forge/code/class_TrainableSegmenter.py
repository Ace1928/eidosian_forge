from skimage.feature import multiscale_basic_features
class TrainableSegmenter:
    """Estimator for classifying pixels.

    Parameters
    ----------
    clf : classifier object, optional
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        ``(m_features, *labels.shape)``. If None,
        :func:`skimage.feature.multiscale_basic_features` is used.

    Methods
    -------
    compute_features
    fit
    predict
    """

    def __init__(self, clf=None, features_func=None):
        if clf is None:
            if has_sklearn:
                self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            else:
                raise ImportError('Please install scikit-learn or pass a classifier instanceto TrainableSegmenter.')
        else:
            self.clf = clf
        self.features_func = features_func

    def compute_features(self, image):
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        self.features = self.features_func(image)

    def fit(self, image, labels):
        """Train classifier using partially labeled (annotated) image.

        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.
        labels : ndarray of ints
            Labeled array of shape compatible with ``image`` (same shape for a
            single-channel image). Labels >= 1 correspond to the training set and
            label 0 to unlabeled pixels to be segmented.
        """
        self.compute_features(image)
        fit_segmenter(labels, self.features, self.clf)

    def predict(self, image):
        """Segment new image using trained internal classifier.

        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.

        Raises
        ------
        NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
        """
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        features = self.features_func(image)
        return predict_segmenter(features, self.clf)