import torch
from torch._export.db.case import export_case

    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    