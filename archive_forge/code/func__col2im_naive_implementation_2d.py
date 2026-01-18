import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
def _col2im_naive_implementation_2d(res, image_shape, kernel_shape, dilations, pads, strides):
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check_2d(res, image_shape, kernel_shape, dilations, new_pads, strides)
    data_col = res.ravel()
    data_im = np.zeros(image_shape, dtype=res.dtype).flatten()
    kernel_h, kernel_w = kernel_shape
    channels_col = kernel_h * kernel_w
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_h, pad_w = new_pads[:, 0]
    height, width = image_shape
    output_height, output_width = image_shape
    height_col = (output_height + new_pads[0, :].sum() - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (output_width + new_pads[1, :].sum() - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    for c_col in range(channels_col):
        w_offset = c_col % kernel_w
        h_offset = c_col // kernel_w % kernel_h
        c_im = c_col // (kernel_h * kernel_w)
        for h_col in range(height_col):
            h_im = h_col * stride_h - pad_h + h_offset * dilation_h
            for w_col in range(width_col):
                w_im = w_col * stride_w - pad_w + w_offset * dilation_w
                if 0 <= h_im < height and 0 <= w_im < width:
                    i_im = (c_im * height + h_im) * width + w_im
                    i_col = (c_col * height_col + h_col) * width_col + w_col
                    if 0 <= i_col < data_col.shape[0]:
                        data_im[i_im] += data_col[i_col]
    return data_im.reshape(image_shape)