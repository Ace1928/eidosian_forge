from __future__ import print_function
import argparse
import re
import mxnet as mx
import caffe_parser
def _convert_pooling_param(param):
    """Convert the pooling layer parameter
    """
    param_string = "pooling_convention='full', "
    if param.global_pooling:
        param_string += 'global_pool=True, kernel=(1,1)'
    else:
        param_string += 'pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)' % (param.pad, param.pad, param.kernel_size, param.kernel_size, param.stride, param.stride)
    if param.pool == 0:
        param_string += ", pool_type='max'"
    elif param.pool == 1:
        param_string += ", pool_type='avg'"
    else:
        raise ValueError('Unknown Pooling Method!')
    return param_string