import numpy as np
import torch
def _prepare_pytorch(x):
    x = x.detach().cpu().numpy()
    return x