from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
def _initialize_variables(test_case, scaffold):
    scaffold.finalize()
    test_case.assertIsNone(scaffold.init_feed_dict)
    test_case.assertIsNone(scaffold.init_fn)
    scaffold.init_op.run()
    scaffold.ready_for_local_init_op.eval()
    scaffold.local_init_op.run()
    scaffold.ready_op.eval()
    test_case.assertIsNotNone(scaffold.saver)