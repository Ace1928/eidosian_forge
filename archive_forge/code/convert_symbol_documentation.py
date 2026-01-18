from __future__ import print_function
import argparse
import re
import mxnet as mx
import caffe_parser
Convert caffe model definition into Symbol

    Parameters
    ----------
    prototxt_fname : str
        Filename of the prototxt file

    Returns
    -------
    Symbol
        Converted Symbol
    tuple
        Input shape
    