import random
import torch
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class ScaledIndexAddBenchmarkBaseline(ScaledIndexAddBenchmark):

    def fw(self) -> None:
        src_scaled = self.src
        if self.scaling is not None:
            src_scaled * self.scaling.unsqueeze(0).unsqueeze(0)
        self.out = self.inp.index_add(dim=0, source=src_scaled, index=self.index, alpha=self.alpha)