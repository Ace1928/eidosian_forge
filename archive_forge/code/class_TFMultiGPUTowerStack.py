from collections import namedtuple, OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from ray.util.debug import log_once
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
@DeveloperAPI
class TFMultiGPUTowerStack:
    """Optimizer that runs in parallel across multiple local devices.

    TFMultiGPUTowerStack automatically splits up and loads training data
    onto specified local devices (e.g. GPUs) with `load_data()`. During a call
    to `optimize()`, the devices compute gradients over slices of the data in
    parallel. The gradients are then averaged and applied to the shared
    weights.

    The data loaded is pinned in device memory until the next call to
    `load_data`, so you can make multiple passes (possibly in randomized order)
    over the same data once loaded.

    This is similar to tf1.train.SyncReplicasOptimizer, but works within a
    single TensorFlow graph, i.e. implements in-graph replicated training:

    https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
    """

    def __init__(self, optimizer=None, devices=None, input_placeholders=None, rnn_inputs=None, max_per_device_batch_size=None, build_graph=None, grad_norm_clipping=None, policy: TFPolicy=None):
        """Initializes a TFMultiGPUTowerStack instance.

        Args:
            policy: The TFPolicy object that this tower stack
                belongs to.
        """
        if policy is None:
            deprecation_warning(old='TFMultiGPUTowerStack(...)', new='TFMultiGPUTowerStack(policy=[Policy])', error=True)
            self.policy = None
            self.optimizers = optimizer
            self.devices = devices
            self.max_per_device_batch_size = max_per_device_batch_size
            self.policy_copy = build_graph
        else:
            self.policy: TFPolicy = policy
            self.optimizers: List[LocalOptimizer] = self.policy._optimizers
            self.devices = self.policy.devices
            self.max_per_device_batch_size = (max_per_device_batch_size or policy.config.get('sgd_minibatch_size', policy.config.get('train_batch_size', 999999))) // len(self.devices)
            input_placeholders = tree.flatten(self.policy._loss_input_dict_no_rnn)
            rnn_inputs = []
            if self.policy._state_inputs:
                rnn_inputs = self.policy._state_inputs + [self.policy._seq_lens]
            grad_norm_clipping = self.policy.config.get('grad_clip')
            self.policy_copy = self.policy.copy
        assert len(self.devices) > 1 or 'gpu' in self.devices[0]
        self.loss_inputs = input_placeholders + rnn_inputs
        shared_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS, scope=tf1.get_variable_scope().name)
        self._batch_index = tf1.placeholder(tf.int32, name='batch_index')
        self._per_device_batch_size = tf1.placeholder(tf.int32, name='per_device_batch_size')
        self._loaded_per_device_batch_size = max_per_device_batch_size
        self._max_seq_len = tf1.placeholder(tf.int32, name='max_seq_len')
        self._loaded_max_seq_len = 1
        device_placeholders = [[] for _ in range(len(self.devices))]
        for t in tree.flatten(self.loss_inputs):
            with tf.device('/cpu:0'):
                splits = tf.split(t, len(self.devices))
            for i, d in enumerate(self.devices):
                device_placeholders[i].append(splits[i])
        self._towers = []
        for tower_i, (device, placeholders) in enumerate(zip(self.devices, device_placeholders)):
            self._towers.append(self._setup_device(tower_i, device, placeholders, len(tree.flatten(input_placeholders))))
        if self.policy.config['_tf_policy_handles_more_than_one_loss']:
            avgs = []
            for i, optim in enumerate(self.optimizers):
                avg = _average_gradients([t.grads[i] for t in self._towers])
                if grad_norm_clipping:
                    clipped = []
                    for grad, _ in avg:
                        clipped.append(grad)
                    clipped, _ = tf.clip_by_global_norm(clipped, grad_norm_clipping)
                    for i, (grad, var) in enumerate(avg):
                        avg[i] = (clipped[i], var)
                avgs.append(avg)
            self._update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS, scope=tf1.get_variable_scope().name)
            for op in shared_ops:
                self._update_ops.remove(op)
            if self._update_ops:
                logger.debug('Update ops to run on apply gradient: {}'.format(self._update_ops))
            with tf1.control_dependencies(self._update_ops):
                self._train_op = tf.group([o.apply_gradients(a) for o, a in zip(self.optimizers, avgs)])
        else:
            avg = _average_gradients([t.grads for t in self._towers])
            if grad_norm_clipping:
                clipped = []
                for grad, _ in avg:
                    clipped.append(grad)
                clipped, _ = tf.clip_by_global_norm(clipped, grad_norm_clipping)
                for i, (grad, var) in enumerate(avg):
                    avg[i] = (clipped[i], var)
            self._update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS, scope=tf1.get_variable_scope().name)
            for op in shared_ops:
                self._update_ops.remove(op)
            if self._update_ops:
                logger.debug('Update ops to run on apply gradient: {}'.format(self._update_ops))
            with tf1.control_dependencies(self._update_ops):
                self._train_op = self.optimizers[0].apply_gradients(avg)
        self.num_grad_updates = 0

    def load_data(self, sess, inputs, state_inputs, num_grad_updates=None):
        """Bulk loads the specified inputs into device memory.

        The shape of the inputs must conform to the shapes of the input
        placeholders this optimizer was constructed with.

        The data is split equally across all the devices. If the data is not
        evenly divisible by the batch size, excess data will be discarded.

        Args:
            sess: TensorFlow session.
            inputs: List of arrays matching the input placeholders, of shape
                [BATCH_SIZE, ...].
            state_inputs: List of RNN input arrays. These arrays have size
                [BATCH_SIZE / MAX_SEQ_LEN, ...].
            num_grad_updates: The lifetime number of gradient updates that the
                policy having collected the data has already undergone.

        Returns:
            The number of tuples loaded per device.
        """
        self.num_grad_updates = num_grad_updates
        if log_once('load_data'):
            logger.info('Training on concatenated sample batches:\n\n{}\n'.format(summarize({'placeholders': self.loss_inputs, 'inputs': inputs, 'state_inputs': state_inputs})))
        feed_dict = {}
        assert len(self.loss_inputs) == len(inputs + state_inputs), (self.loss_inputs, inputs, state_inputs)
        if len(state_inputs) > 0:
            smallest_array = state_inputs[0]
            seq_len = len(inputs[0]) // len(state_inputs[0])
            self._loaded_max_seq_len = seq_len
        else:
            smallest_array = inputs[0]
            self._loaded_max_seq_len = 1
        sequences_per_minibatch = self.max_per_device_batch_size // self._loaded_max_seq_len * len(self.devices)
        if sequences_per_minibatch < 1:
            logger.warning('Target minibatch size is {}, however the rollout sequence length is {}, hence the minibatch size will be raised to {}.'.format(self.max_per_device_batch_size, self._loaded_max_seq_len, self._loaded_max_seq_len * len(self.devices)))
            sequences_per_minibatch = 1
        if len(smallest_array) < sequences_per_minibatch:
            sequences_per_minibatch = _make_divisible_by(len(smallest_array), len(self.devices))
        if log_once('data_slicing'):
            logger.info('Divided {} rollout sequences, each of length {}, among {} devices.'.format(len(smallest_array), self._loaded_max_seq_len, len(self.devices)))
        if sequences_per_minibatch < len(self.devices):
            raise ValueError('Must load at least 1 tuple sequence per device. Try increasing `sgd_minibatch_size` or reducing `max_seq_len` to ensure that at least one sequence fits per device.')
        self._loaded_per_device_batch_size = sequences_per_minibatch // len(self.devices) * self._loaded_max_seq_len
        if len(state_inputs) > 0:
            state_inputs = [_make_divisible_by(arr, sequences_per_minibatch) for arr in state_inputs]
            inputs = [arr[:len(state_inputs[0]) * seq_len] for arr in inputs]
            assert len(state_inputs[0]) * seq_len == len(inputs[0]), (len(state_inputs[0]), sequences_per_minibatch, seq_len, len(inputs[0]))
            for ph, arr in zip(self.loss_inputs, inputs + state_inputs):
                feed_dict[ph] = arr
            truncated_len = len(inputs[0])
        else:
            truncated_len = 0
            for ph, arr in zip(self.loss_inputs, inputs):
                truncated_arr = _make_divisible_by(arr, sequences_per_minibatch)
                feed_dict[ph] = truncated_arr
                if truncated_len == 0:
                    truncated_len = len(truncated_arr)
        sess.run([t.init_op for t in self._towers], feed_dict=feed_dict)
        self.num_tuples_loaded = truncated_len
        samples_per_device = truncated_len // len(self.devices)
        assert samples_per_device > 0, 'No data loaded?'
        assert samples_per_device % self._loaded_per_device_batch_size == 0
        return samples_per_device

    def optimize(self, sess, batch_index):
        """Run a single step of SGD.

        Runs a SGD step over a slice of the preloaded batch with size given by
        self._loaded_per_device_batch_size and offset given by the batch_index
        argument.

        Updates shared model weights based on the averaged per-device
        gradients.

        Args:
            sess: TensorFlow session.
            batch_index: Offset into the preloaded data. This value must be
                between `0` and `tuples_per_device`. The amount of data to
                process is at most `max_per_device_batch_size`.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        """
        feed_dict = {self._batch_index: batch_index, self._per_device_batch_size: self._loaded_per_device_batch_size, self._max_seq_len: self._loaded_max_seq_len}
        for tower in self._towers:
            feed_dict.update(tower.loss_graph.extra_compute_grad_feed_dict())
        fetches = {'train': self._train_op}
        for tower_num, tower in enumerate(self._towers):
            tower_fetch = tower.loss_graph._get_grad_and_stats_fetches()
            fetches['tower_{}'.format(tower_num)] = tower_fetch
        return sess.run(fetches, feed_dict=feed_dict)

    def get_device_losses(self):
        return [t.loss_graph for t in self._towers]

    def _setup_device(self, tower_i, device, device_input_placeholders, num_data_in):
        assert num_data_in <= len(device_input_placeholders)
        with tf.device(device):
            with tf1.name_scope(TOWER_SCOPE_NAME + f'_{tower_i}'):
                device_input_batches = []
                device_input_slices = []
                for i, ph in enumerate(device_input_placeholders):
                    current_batch = tf1.Variable(ph, trainable=False, validate_shape=False, collections=[])
                    device_input_batches.append(current_batch)
                    if i < num_data_in:
                        scale = self._max_seq_len
                        granularity = self._max_seq_len
                    else:
                        scale = self._max_seq_len
                        granularity = 1
                    current_slice = tf.slice(current_batch, [self._batch_index // scale * granularity] + [0] * len(ph.shape[1:]), [self._per_device_batch_size // scale * granularity] + [-1] * len(ph.shape[1:]))
                    current_slice.set_shape(ph.shape)
                    device_input_slices.append(current_slice)
                graph_obj = self.policy_copy(device_input_slices)
                device_grads = graph_obj.gradients(self.optimizers, graph_obj._losses)
            return _Tower(tf.group(*[batch.initializer for batch in device_input_batches]), device_grads, graph_obj)