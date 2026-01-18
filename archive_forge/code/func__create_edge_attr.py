import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def _create_edge_attr(self, src, dst, transition):
    label_pos = 'label'
    attr = {}
    if src in self._cluster_states:
        attr['ltail'] = 'cluster_' + src
        src_name = src + '_anchor'
        label_pos = 'headlabel'
    else:
        src_name = src
    if dst in self._cluster_states:
        if not src.startswith(dst):
            attr['lhead'] = 'cluster_' + dst
            label_pos = 'taillabel' if label_pos.startswith('l') else 'label'
        dst_name = dst + '_anchor'
    else:
        dst_name = dst
    if 'ltail' in attr and dst_name.startswith(attr['ltail'][8:]):
        del attr['ltail']
    attr[label_pos] = self._transition_label(transition)
    attr['label_pos'] = label_pos
    attr['source'] = src_name
    attr['dest'] = dst_name
    return attr