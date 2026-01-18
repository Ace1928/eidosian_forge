import numpy as np
from onnx.reference.ops._op_common_pool import CommonPool
def _max_pool_1d(self, x, auto_pad, ceil_mode, dilations, kernel_shape, new_pads, storage_order, strides, output_spatial_shape):
    global_pooling = False
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]
    height = x_dims[2]
    pooled_height = y_dims[2]
    total_channels = x_dims[0] * channels
    stride_h = 1 if global_pooling else strides[0]
    x_step = height
    y_step = pooled_height
    dilation_h = dilations[0]
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()

    def iteration(c):
        x_d = c * x_step
        y_d = c * y_step
        i_d = c * y_step
        for ph in range(pooled_height):
            hstart = ph * stride_h - new_pads[0, 0]
            hend = hstart + kernel_shape[0] * dilation_h
            Yh = None
            h_index = -1
            for h in range(hstart, hend, dilation_h):
                if h < 0 or h >= height:
                    continue
                if Yh is None or X_data[x_d + h] > Yh:
                    Yh = X_data[x_d + h]
                    h_index = h
            Y_data[y_d + ph] = Yh
            I_data[i_d + ph] = c * x_step + h_index
    for c in range(total_channels):
        iteration(c)
    if len(self.output) == 1:
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))