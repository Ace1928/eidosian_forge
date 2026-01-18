import copy
import functools
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import strutils
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
import glanceclient.v1.images
def _image_show(image, human_readable=False, max_column_width=80):
    info = copy.deepcopy(image._info)
    if human_readable:
        info['size'] = utils.make_size_human_readable(info['size'])
    for k, v in info.pop('properties').items():
        info["Property '%s'" % k] = v
    utils.print_dict(info, max_column_width=max_column_width)