from warnings import simplefilter
import pandas as pd
import sklearn
import wandb
from wandb.sklearn import calculate, utils
Measures & plots silhouette coefficients.

    Silhouette coefficients near +1 indicate that the sample is far away from
    the neighboring clusters. A value near 0 indicates that the sample is on or
    very close to the decision boundary between two neighboring clusters and
    negative values indicate that the samples might have been assigned to the wrong cluster.

    Should only be called with a fitted clusterer (otherwise an error is thrown).

    Please note this function fits the model on the training set when called.

    Arguments:
        model: (clusterer) Takes in a fitted clusterer.
        X: (arr) Training set features.
        cluster_labels: (list) Names for cluster labels. Makes plots easier to read
                               by replacing cluster indexes with corresponding names.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_silhouette(model, X_train, ["spam", "not spam"])
    ```
    