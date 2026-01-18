from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
def _update_mems(self, hids, mems, mlen, qlen):
    if mems is None:
        return None
    assert len(hids) == len(mems), 'len(hids) != len(mems)'
    new_mems = []
    end_idx = mlen + tf.math.maximum(0, qlen)
    beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
    for i in range(len(hids)):
        mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
        cat = tf.concat([mems[i], hids[i]], axis=0)
        tf.stop_gradient(cat)
        new_mems.append(cat[beg_idx:end_idx])
    return new_mems