from __future__ import print_function
import argparse
import re
import mxnet as mx
import caffe_parser
def _convert_conv_param(param):
    """
    Convert convolution layer parameter from Caffe to MXNet
    """
    param_string = 'num_filter=%d' % param.num_output
    pad_w = 0
    pad_h = 0
    if isinstance(param.pad, int):
        pad = param.pad
        param_string += ', pad=(%d, %d)' % (pad, pad)
    elif len(param.pad) > 0:
        pad = param.pad[0]
        param_string += ', pad=(%d, %d)' % (pad, pad)
    else:
        if isinstance(param.pad_w, int):
            pad_w = param.pad_w
        if isinstance(param.pad_h, int):
            pad_h = param.pad_h
        param_string += ', pad=(%d, %d)' % (pad_h, pad_w)
    if isinstance(param.kernel_size, int):
        kernel_size = param.kernel_size
        param_string += ', kernel=(%d,%d)' % (kernel_size, kernel_size)
    elif len(param.kernel_size) > 0:
        kernel_size = param.kernel_size[0]
        param_string += ', kernel=(%d,%d)' % (kernel_size, kernel_size)
    else:
        assert isinstance(param.kernel_w, int)
        kernel_w = param.kernel_w
        assert isinstance(param.kernel_h, int)
        kernel_h = param.kernel_h
        param_string += ', kernel=(%d,%d)' % (kernel_h, kernel_w)
    stride = 1
    if isinstance(param.stride, int):
        stride = param.stride
    else:
        stride = 1 if len(param.stride) == 0 else param.stride[0]
    param_string += ', stride=(%d,%d)' % (stride, stride)
    dilate = 1
    if hasattr(param, 'dilation'):
        if isinstance(param.dilation, int):
            dilate = param.dilation
        else:
            dilate = 1 if len(param.dilation) == 0 else param.dilation[0]
    param_string += ', no_bias=%s' % (not param.bias_term)
    if dilate > 1:
        param_string += ', dilate=(%d, %d)' % (dilate, dilate)
    if isinstance(param.group, int):
        if param.group != 1:
            param_string += ', num_group=%d' % param.group
    return param_string