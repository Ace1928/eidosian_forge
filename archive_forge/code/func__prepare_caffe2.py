import numpy as np
import torch
def _prepare_caffe2(x):
    from caffe2.python import workspace
    x = workspace.FetchBlob(x)
    return x