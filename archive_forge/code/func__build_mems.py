from functools import lru_cache
import torch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .modules import MemNN, opt_to_kwargs
def _build_mems(self, mems):
    """
        Build memory tensors.

        During building, will add time features to the memories if enabled.

        :param mems:
            list of length batchsize containing inner lists of 1D tensors
            containing the individual memories for each row in the batch.

        :returns:
            3d padded tensor of memories (bsz x num_mems x seqlen)
        """
    if mems is None:
        return None
    bsz = len(mems)
    if bsz == 0:
        return None
    num_mems = max((len(mem) for mem in mems))
    if num_mems == 0 or self.memsize <= 0:
        return None
    elif num_mems > self.memsize:
        num_mems = self.memsize
        mems = [mem[-self.memsize:] for mem in mems]
    try:
        seqlen = max((len(m) for mem in mems for m in mem))
        if self.use_time_features:
            seqlen += 1
    except ValueError:
        return None
    padded = torch.LongTensor(bsz, num_mems, seqlen).fill_(0)
    for i, mem in enumerate(mems):
        tf_offset = len(mem) - 1
        for j, m in enumerate(mem):
            padded[i, j, :len(m)] = m
            if self.use_time_features:
                padded[i, j, -1] = self.dict[self._time_feature(tf_offset - j)]
    if self.use_cuda:
        padded = padded.cuda()
    return padded