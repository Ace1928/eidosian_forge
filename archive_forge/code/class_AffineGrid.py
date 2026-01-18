import numpy as np
from onnx.reference.op_run import OpRun
class AffineGrid(OpRun):

    def _run(self, theta, size, align_corners=None):
        align_corners = align_corners or self.align_corners
        _, _, *data_size = size
        original_grid = construct_original_grid(data_size, align_corners)
        grid = apply_affine_transform(theta, original_grid)
        return (grid,)