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
def _cluster_markers(markers, tissues, marker_labels, tissue_labels, marker_groups_order, s, c):
    markers_order = []
    for marker_group in marker_groups_order:
        if len(marker_group) > 1:
            marker_names = markers[marker_group]
            marker_features = []
            for marker in marker_names:
                marker_idx = np.array(marker_labels) == marker
                if tissues is not None:
                    marker_idx = marker_idx & (tissue_labels == tissues[marker_group[0]])
                marker_features.append(np.concatenate([s[marker_idx], c[marker_idx]]))
            marker_features = np.array(marker_features)
            marker_features = marker_features / np.sqrt(np.sum(marker_features ** 2))
            marker_group_order = hierarchy.leaves_list(hierarchy.linkage(marker_features))
            markers_order.append(marker_group[marker_group_order])
        else:
            markers_order.append(marker_group)
    markers_order = np.concatenate(markers_order)
    return markers_order