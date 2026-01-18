import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def _process_layer_parameters(layer):
    logging.debug('processing layer %s of type %s', layer.name, layer.type)
    normalized_layer_name = re.sub('[-/]', '_', layer.name)
    if layer.name in caffe_net.params and layer.type in ['Convolution', 'InnerProduct', 'Deconvolution']:
        has_bias = len(caffe_net.params[layer.name]) > 1
        mx_name_weight = '{}_weight'.format(normalized_layer_name)
        mx_beta = arg_params[mx_name_weight].asnumpy()
        if layer.type == 'Convolution' and compare_layers_from_nets.is_first_convolution:
            compare_layers_from_nets.is_first_convolution = False
            if mx_beta.shape[1] == 3 or mx_beta.shape[1] == 4:
                mx_beta[:, [0, 2], :, :] = mx_beta[:, [2, 0], :, :]
        caf_beta = caffe_net.params[layer.name][0].data
        _compare_blob(caf_beta, mx_beta, layer.name, mx_name_weight, 'weight', '')
        if has_bias:
            mx_name_bias = '{}_bias'.format(normalized_layer_name)
            mx_gamma = arg_params[mx_name_bias].asnumpy()
            caf_gamma = caffe_net.params[layer.name][1].data
            _compare_blob(caf_gamma, mx_gamma, layer.name, mx_name_bias, 'bias', '')
    elif layer.name in caffe_net.params and layer.type == 'Scale':
        if 'scale' in normalized_layer_name:
            bn_name = normalized_layer_name.replace('scale', 'bn')
        elif 'sc' in normalized_layer_name:
            bn_name = normalized_layer_name.replace('sc', 'bn')
        else:
            assert False, 'Unknown name convention for bn/scale'
        beta_name = '{}_beta'.format(bn_name)
        gamma_name = '{}_gamma'.format(bn_name)
        mx_beta = arg_params[beta_name].asnumpy()
        caf_beta = caffe_net.params[layer.name][1].data
        _compare_blob(caf_beta, mx_beta, layer.name, beta_name, 'mov_mean', '')
        mx_gamma = arg_params[gamma_name].asnumpy()
        caf_gamma = caffe_net.params[layer.name][0].data
        _compare_blob(caf_gamma, mx_gamma, layer.name, gamma_name, 'mov_var', '')
    elif layer.name in caffe_net.params and layer.type == 'BatchNorm':
        mean_name = '{}_moving_mean'.format(normalized_layer_name)
        var_name = '{}_moving_var'.format(normalized_layer_name)
        caf_rescale_factor = caffe_net.params[layer.name][2].data
        mx_mean = aux_params[mean_name].asnumpy()
        caf_mean = caffe_net.params[layer.name][0].data / caf_rescale_factor
        _compare_blob(caf_mean, mx_mean, layer.name, mean_name, 'mean', '')
        mx_var = aux_params[var_name].asnumpy()
        caf_var = caffe_net.params[layer.name][1].data / caf_rescale_factor
        _compare_blob(caf_var, mx_var, layer.name, var_name, 'var', 'expect 1e-04 change due to cudnn eps')
    elif layer.type in ['Input', 'Pooling', 'ReLU', 'Eltwise', 'Softmax', 'LRN', 'Concat', 'Dropout', 'Crop']:
        pass
    else:
        warnings.warn('No handling for layer %s of type %s, should we ignore it?', layer.name, layer.type)