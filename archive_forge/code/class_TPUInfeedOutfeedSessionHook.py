from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
class TPUInfeedOutfeedSessionHook(tf.compat.v1.train.SessionRunHook):
    """A Session hook setting up the TPU initialization, infeed, and outfeed.

  This hook does two major things:
  1. initialize and shutdown TPU system.
  2. launch and join the threads for infeed enqueue and (optional) outfeed
     dequeue.
  """

    def __init__(self, ctx, enqueue_ops, dequeue_ops, tpu_compile_op, run_infeed_loop_on_coordinator=True, rendezvous=None, master=None, session_config=None, tpu_init_ops=None, outfeed_every_n_steps=1):
        self._master_job = ctx.master_job
        self._enqueue_ops = enqueue_ops
        self._dequeue_ops = dequeue_ops
        self._rendezvous = rendezvous
        self._master = master
        self._session_config = session_config
        self._init_ops = list(tpu_init_ops or [])
        if ctx.embedding_config is None:
            self._embedding_layer_config = None
        else:
            self._embedding_layer_config = ctx.embedding_config.tpu_embedding.config_proto
        self._run_infeed_loop_on_coordinator = run_infeed_loop_on_coordinator
        self._initial_infeed_sleep_secs = ctx.config.tpu_config.initial_infeed_sleep_secs
        self._tpu_compile_op = tpu_compile_op
        if ctx.model_parallelism_enabled and ctx.config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.BROADCAST:
            self._should_initialize_tpu = False
        else:
            self._should_initialize_tpu = True
        self._outfeed_every_n_steps = outfeed_every_n_steps

    def begin(self):
        tf.compat.v1.logging.info('TPU job name %s', self._master_job)
        self._iterations_per_loop_var = _create_or_get_iterations_per_loop()
        if self._should_initialize_tpu:
            self._finalize_ops = [tf.compat.v1.tpu.shutdown_system(job=self._master_job)]
        else:
            self._finalize_ops = []
        summary_writer_init_ops = summary_ops_v2.summary_writer_initializer_op()
        self._init_ops.extend(summary_writer_init_ops)
        for op in summary_writer_init_ops:
            self._finalize_ops.append(summary_ops_v2.legacy_raw_flush(writer=op.inputs[0]))

    def _run_infeed(self, queue_ctx, session):
        tf.compat.v1.logging.info('Starting infeed thread controller.')
        if self._initial_infeed_sleep_secs:
            tf.compat.v1.logging.info('Infeed thread sleeping for %d seconds.', self._initial_infeed_sleep_secs)
            time.sleep(self._initial_infeed_sleep_secs)
            tf.compat.v1.logging.info('Infeed thread starting after sleep')
        with self._rendezvous.catch_errors(source='infeed', session=session):
            if self._run_infeed_loop_on_coordinator:
                for count, steps in enumerate(queue_ctx.read_iteration_counts()):
                    for i in xrange(steps):
                        tf.compat.v1.logging.debug('Infeed enqueue for iteration (%d, %d)', count, i)
                        session.run(self._enqueue_ops)
            else:
                for _ in queue_ctx.read_iteration_counts():
                    session.run(self._enqueue_ops)
            tf.compat.v1.logging.info('Infeed thread finished, shutting down.')

    def _run_outfeed(self, queue_ctx, session):
        tf.compat.v1.logging.info('Starting outfeed thread controller.')
        status_logger = PeriodicLogger(seconds=60)
        with self._rendezvous.catch_errors(source='outfeed', session=session):
            for count, steps in enumerate(queue_ctx.read_iteration_counts()):
                step_counter = 0
                for i in xrange(steps):
                    tf.compat.v1.logging.debug('Outfeed dequeue for iteration (%d, %d)', count, i)
                    if step_counter % self._outfeed_every_n_steps == 0:
                        session.run(self._dequeue_ops)
                    step_counter += 1
                    status_logger.log('Outfeed finished for iteration (%d, %d)', count, i)
            tf.compat.v1.logging.info('Outfeed thread finished, shutting down.')

    def _create_infeed_controller(self, name, target, args):
        return _OpQueueContext(name=name, target=target, args=args)

    def _assertCompilationSucceeded(self, result, coord):
        proto = tpu_compilation_result.CompilationResultProto()
        proto.ParseFromString(result)
        if proto.status_error_message:
            tf.compat.v1.logging.error('Compilation failed: {}'.format(proto.status_error_message))
            coord.request_stop()
        else:
            tf.compat.v1.logging.info('Compilation succeeded')

    def after_create_session(self, session, coord):
        if self._should_initialize_tpu:
            tf.compat.v1.logging.info('Init TPU system')
            start = time.time()
            with tf.Graph().as_default():
                with tf.compat.v1.Session(self._master, config=self._session_config) as sess:
                    sess.run(tf.compat.v1.tpu.initialize_system(job=self._master_job, embedding_config=self._embedding_layer_config))
            tf.compat.v1.logging.info('Initialized TPU in %d seconds', time.time() - start)
        session.run(self._init_ops, options=tf.compat.v1.RunOptions(timeout_in_ms=30 * 60 * 1000))
        if os.environ.get('TPU_SPLIT_COMPILE_AND_EXECUTE', '') == '1':
            tf.compat.v1.logging.info('Compiling user program: this may take a while...')
            self._assertCompilationSucceeded(session.run(self._tpu_compile_op), coord)
        self._infeed_controller = self._create_infeed_controller(name='InfeedController', target=self._run_infeed, args=(session,))
        self._outfeed_controller = _OpQueueContext(name='OutfeedController', target=self._run_outfeed, args=(session,))
        watchdog_timeout = int(os.environ.get('TF_TPU_WATCHDOG_TIMEOUT', '0'))
        if watchdog_timeout > 0:
            session_support.start_worker_watchdog(session, shutdown_timeout=watchdog_timeout)

    def before_run(self, run_context):
        iterations = run_context.session.run(self._iterations_per_loop_var)
        tf.compat.v1.logging.info('Enqueue next (%d) batch(es) of data to infeed.', iterations)
        self._infeed_controller.send_next_batch_signal(iterations)
        tf.compat.v1.logging.info('Dequeue next (%d) batch(es) of data from outfeed.', iterations)
        self._outfeed_controller.send_next_batch_signal(iterations)

    def end(self, session):
        tf.compat.v1.logging.info('Stop infeed thread controller')
        self._infeed_controller.join()
        self._rendezvous.record_done('infeed')
        tf.compat.v1.logging.info('Stop output thread controller')
        self._outfeed_controller.join()
        self._rendezvous.record_done('outfeed')
        tf.compat.v1.logging.info('Shutdown TPU system.')
        session.run(self._finalize_ops)