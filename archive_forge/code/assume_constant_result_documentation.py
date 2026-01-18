import torch
import torch._dynamo as torchdynamo
from torch._export.db.case import export_case

    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    