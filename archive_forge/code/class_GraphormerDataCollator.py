from typing import Any, Dict, List, Mapping
import numpy as np
import torch
from ...utils import is_cython_available, requires_backends
class GraphormerDataCollator:

    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        if not is_cython_available():
            raise ImportError('Graphormer preprocessing needs Cython (pyximport)')
        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if self.on_the_fly_processing:
            features = [preprocess_item(i) for i in features]
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}
        max_node_num = max((len(i['input_nodes']) for i in features))
        node_feat_size = len(features[0]['input_nodes'][0])
        edge_feat_size = len(features[0]['attn_edge_type'][0][0])
        max_dist = max((len(i['input_edges'][0][0]) for i in features))
        edge_input_size = len(features[0]['input_edges'][0][0][0])
        batch_size = len(features)
        batch['attn_bias'] = torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch['attn_edge_type'] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch['spatial_pos'] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch['in_degree'] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch['input_nodes'] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch['input_edges'] = torch.zeros(batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long)
        for ix, f in enumerate(features):
            for k in ['attn_bias', 'attn_edge_type', 'spatial_pos', 'in_degree', 'input_nodes', 'input_edges']:
                f[k] = torch.tensor(f[k])
            if len(f['attn_bias'][1:, 1:][f['spatial_pos'] >= self.spatial_pos_max]) > 0:
                f['attn_bias'][1:, 1:][f['spatial_pos'] >= self.spatial_pos_max] = float('-inf')
            batch['attn_bias'][ix, :f['attn_bias'].shape[0], :f['attn_bias'].shape[1]] = f['attn_bias']
            batch['attn_edge_type'][ix, :f['attn_edge_type'].shape[0], :f['attn_edge_type'].shape[1], :] = f['attn_edge_type']
            batch['spatial_pos'][ix, :f['spatial_pos'].shape[0], :f['spatial_pos'].shape[1]] = f['spatial_pos']
            batch['in_degree'][ix, :f['in_degree'].shape[0]] = f['in_degree']
            batch['input_nodes'][ix, :f['input_nodes'].shape[0], :] = f['input_nodes']
            batch['input_edges'][ix, :f['input_edges'].shape[0], :f['input_edges'].shape[1], :f['input_edges'].shape[2], :] = f['input_edges']
        batch['out_degree'] = batch['in_degree']
        sample = features[0]['labels']
        if len(sample) == 1:
            if isinstance(sample[0], float):
                batch['labels'] = torch.from_numpy(np.concatenate([i['labels'] for i in features]))
            else:
                batch['labels'] = torch.from_numpy(np.concatenate([i['labels'] for i in features]))
        else:
            batch['labels'] = torch.from_numpy(np.stack([i['labels'] for i in features], axis=0))
        return batch