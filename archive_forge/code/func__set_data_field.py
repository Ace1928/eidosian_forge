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
def _set_data_field(fields, args):
    if 'location' not in fields and 'copy_from' not in fields:
        fields['data'] = utils.get_data_file(args)