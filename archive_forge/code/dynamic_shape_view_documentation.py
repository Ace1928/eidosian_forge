import torch
from torch._export.db.case import export_case

    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    