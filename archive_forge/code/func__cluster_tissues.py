from .. import select
from .. import stats
from .. import utils
from .tools import label_axis
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import shift_ticklabels
from .utils import show
from .utils import temp_fontsize
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
def _cluster_tissues(tissue_names, cluster_names, tissue_labels, cluster_labels, s, c):
    tissue_features = []
    for tissue in tissue_names:
        tissue_data = []
        for cluster in cluster_names:
            tissue_cluster_idx = np.where((np.array(tissue_labels) == tissue) & (np.array(cluster_labels) == cluster))
            tissue_data.append(np.vstack([s[tissue_cluster_idx], c[tissue_cluster_idx]]).mean(axis=1))
        tissue_features.append(np.concatenate(tissue_data))
    tissue_features = np.array(tissue_features)
    tissue_features = tissue_features / np.sqrt(np.sum(tissue_features ** 2))
    tissues_order = hierarchy.leaves_list(hierarchy.linkage(tissue_features))
    return tissues_order