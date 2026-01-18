import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def _process_layer_output(caffe_blob_name):
    logging.debug('processing blob %s', caffe_blob_name)
    if caffe_blob_name not in top_to_layers:
        return
    caf_blob = caffe_net.blobs[caffe_blob_name].data
    if caffe_blob_name == 'data':
        if caf_blob.shape[1] == 3 or caf_blob.shape[1] == 4:
            caf_blob[:, [0, 2], :, :] = caf_blob[:, [2, 0], :, :]
        mx_name = 'data'
    else:
        last_layer_name = top_to_layers[caffe_blob_name][-1]
        normalized_last_layer_name = re.sub('[-/]', '_', last_layer_name)
        mx_name = '{}_output'.format(normalized_last_layer_name)
        if 'scale' in mx_name:
            mx_name = mx_name.replace('scale', 'bn')
        elif 'sc' in mx_name:
            mx_name = mx_name.replace('sc', 'bn')
    if mx_name not in exe.output_dict:
        logging.error('mxnet blob %s is missing, time to extend the compare tool..', mx_name)
        return
    mx_blob = exe.output_dict[mx_name].asnumpy()
    _compare_blob(caf_blob, mx_blob, caffe_blob_name, mx_name, 'output', '')
    return