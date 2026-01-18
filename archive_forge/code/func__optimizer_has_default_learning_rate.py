from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from absl import logging
import six
import tensorflow as tf
def _optimizer_has_default_learning_rate(opt):
    signature = inspect.getfullargspec(opt.__init__)
    default_name_to_value = dict(zip(signature.args[::-1], signature.defaults))
    for name in signature.kwonlyargs:
        if name in signature.kwonlydefaults:
            default_name_to_value[name] = signature.kwonlydefaults[name]
    return 'learning_rate' in default_name_to_value