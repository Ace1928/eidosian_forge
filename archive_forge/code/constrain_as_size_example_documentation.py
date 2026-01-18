import torch
from torch._export.db.case import export_case

    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at constrain_as_value and constrain_as_size APIs
    constrain_as_size is used for values that NEED to be used for constructing
    tensor.
    