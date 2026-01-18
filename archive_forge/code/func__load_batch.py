import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq
import orjson
from ochat.training_deepspeed.multipack_sampler import MultipackDistributedSampler
def _load_batch(self, indices):
    batch = {k: v[indices] for k, v in self.dataset.items()}
    batch = {k: np.concatenate(batch[k], axis=0) for k in self.BATCH_KEYS.keys()}
    total_seqlen = batch['nz_input_ids'].size
    pad_len = _find_multiple(total_seqlen, self.PAD_MULTIPLE) - total_seqlen
    if pad_len > 0:
        assert pad_len < self.PAD_MULTIPLE
        padding_specs = {'seqlens': (1, pad_len), 'nz_input_ids': (pad_len, self.PAD_ID), 'nz_position_ids': (pad_len, 0), 'nz_shifted_label_ids': (pad_len, self.PAD_ID), 'nz_shifted_loss_weights': (pad_len, 0)}
        for k, pad_spec in padding_specs.items():
            batch[k] = np.concatenate((batch[k], np.full(*pad_spec, dtype=batch[k].dtype)), axis=0)
    batch_tensor = {}
    for k, dtype in self.BATCH_KEYS.items():
        batch_tensor[k] = torch.from_numpy(batch[k]).to(dtype)
    batch_tensor['cu_seqlens'] = torch.nn.functional.pad(batch_tensor['seqlens'].cumsum(-1, dtype=torch.int32), (1, 0))
    batch_info = {'max_seqlen': torch.max(batch_tensor['seqlens']).item()}
    del batch_tensor['seqlens']
    return (batch_tensor, batch_info)