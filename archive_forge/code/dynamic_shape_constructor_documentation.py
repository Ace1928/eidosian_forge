import torch
from torch._export.db.case import export_case

    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    