import time
from warnings import simplefilter
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
import wandb
def _clone_and_score_clusterer(clusterer, X, n_clusters):
    start = time.time()
    clusterer = clone(clusterer)
    setattr(clusterer, 'n_clusters', n_clusters)
    return (clusterer.fit(X).score(X), time.time() - start)