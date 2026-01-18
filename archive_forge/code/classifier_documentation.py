from warnings import simplefilter
import numpy as np
from sklearn import naive_bayes
import wandb
import wandb.plots
from wandb.sklearn import calculate, utils
from . import shared
Log a plot depicting how well-calibrated the predicted probabilities of a classifier are.

    Also suggests how to calibrate an uncalibrated classifier. Compares estimated predicted
    probabilities by a baseline logistic regression model, the model passed as
    an argument, and by both its isotonic calibration and sigmoid calibrations.
    The closer the calibration curves are to a diagonal the better.
    A sine wave like curve represents an overfitted classifier, while a cosine
    wave like curve represents an underfitted classifier.
    By training isotonic and sigmoid calibrations of the model and comparing
    their curves we can figure out whether the model is over or underfitting and
    if so which calibration (sigmoid or isotonic) might help fix this.
    For more details, see https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html.

    Should only be called with a fitted classifer (otherwise an error is thrown).

    Please note this function fits variations of the model on the training set when called.

    Arguments:
        clf: (clf) Takes in a fitted classifier.
        X: (arr) Training set features.
        y: (arr) Training set labels.
        model_name: (str) Model name. Defaults to 'Classifier'

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_calibration_curve(clf, X, y, "RandomForestClassifier")
    ```
    