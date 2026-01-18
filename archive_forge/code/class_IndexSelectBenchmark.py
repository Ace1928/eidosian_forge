import random
import torch
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class IndexSelectBenchmark:

    def __init__(self, dtype, batches, D, keep_ratio, bw: bool) -> None:
        dtype_str = DTYPE2STR.get(dtype, dtype)
        self.sub_label = f'{dtype_str} D={D} batches={batches} keep={keep_ratio}'
        self.label = 'index_select'
        indices = []
        sources = []
        for B, seqlen in batches:
            index = [i for i in range(B)]
            random.Random(B).shuffle(index)
            indices.append(torch.zeros(index[int(keep_ratio * B)], dtype=torch.int64, device='cuda'))
            source_i = torch.randn([B, seqlen * D], dtype=dtype, device='cuda', requires_grad=bw)
            sources.append(source_i)
        self.indices, self.sources = (indices, sources)
        self.out = torch.Tensor()

    def fw(self) -> None:
        self.out = xops.index_select_cat(self.sources, self.indices)

    def bw(self):
        for src in self.sources:
            src.grad = None
        self.out.backward(self.out, retain_graph=True)