import sys
from typing import Any, Dict, Type
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2
import xformers.ops as xops
def _setup_test(functions, fw: bool=False, bw: bool=False, cuda_graph: bool=True, **kwargs):
    for k, benchmark_cls in functions.items():
        benchmark_object = benchmark_cls(**kwargs, bw=bw)
        label = benchmark_object.label
        label += 'fw' if fw else ''
        label += 'bw' if bw else ''

        def run_one():
            if fw:
                benchmark_object.fw()
            if bw:
                benchmark_object.bw()
        if cuda_graph:
            run_one()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run_one()

            def run_one():
                g.replay()
        yield benchmark.Timer(stmt='fn()', globals={'fn': run_one}, label=label, description=k, sub_label=benchmark_object.sub_label)