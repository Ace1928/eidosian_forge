import os
import time
import logging
import warnings
from collections import namedtuple
import numpy as np
from . import io
from . import ndarray as nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t
from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position
def _train_multi_device(symbol, ctx, arg_names, param_names, aux_names, arg_params, aux_params, begin_epoch, end_epoch, epoch_size, optimizer, kvstore, update_on_kvstore, train_data, eval_data=None, eval_metric=None, epoch_end_callback=None, batch_end_callback=None, logger=None, work_load_list=None, monitor=None, eval_end_callback=None, eval_batch_end_callback=None, sym_gen=None):
    """Internal training function on multiple devices.
    This function will also work for single device as well.

    Parameters
    ----------
    symbol : Symbol
        The network configuration.
    ctx : list of Context
        The training devices.
    arg_names: list of str
        Name of all arguments of the network.
    param_names: list of str
        Name of all trainable parameters of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    begin_epoch : int
        The begining training epoch.
    end_epoch : int
        The end training epoch.
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ``ceil(num_train_examples / batch_size)``.
    optimizer : Optimizer
        The optimization algorithm
    train_data : DataIter
        Training data iterator.
    eval_data : DataIter
        Validation data iterator.
    eval_metric : EvalMetric
        An evaluation function or a list of evaluation functions.
    epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
        A callback that is invoked at end of each epoch.
        This can be used to checkpoint model each epoch.
    batch_end_callback : callable(BatchEndParams)
        A callback that is invoked at end of each batch.
        This can be used to measure speed, get result from evaluation metric. etc.
    kvstore : KVStore
        The KVStore.
    update_on_kvstore : bool
        Whether or not perform weight updating on kvstore.
    logger : logging logger
        When not specified, default logger will be used.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ``ctx``.
    monitor : Monitor, optional
        Monitor installed to executor,
        for monitoring outputs, weights, and gradients for debugging.
    Notes
    -----
    - This function will inplace update the NDArrays in `arg_params` and `aux_states`.
    """
    if logger is None:
        logger = logging
    executor_manager = DataParallelExecutorManager(symbol=symbol, sym_gen=sym_gen, ctx=ctx, train_data=train_data, param_names=param_names, arg_names=arg_names, aux_names=aux_names, work_load_list=work_load_list, logger=logger)
    if monitor:
        executor_manager.install_monitor(monitor)
    executor_manager.set_params(arg_params, aux_params)
    if not update_on_kvstore:
        updater = get_updater(optimizer)
    else:
        kvstore.set_optimizer(optimizer)
    if kvstore:
        _initialize_kvstore(kvstore=kvstore, param_arrays=executor_manager.param_arrays, arg_params=arg_params, param_names=executor_manager.param_names, update_on_kvstore=update_on_kvstore)
    train_data.reset()
    for epoch in range(begin_epoch, end_epoch):
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        while True:
            do_reset = True
            for data_batch in train_data:
                executor_manager.load_data_batch(data_batch)
                if monitor is not None:
                    monitor.tic()
                executor_manager.forward(is_train=True)
                executor_manager.backward()
                if update_on_kvstore:
                    if 'nccl' in kvstore.type:
                        _update_params_on_kvstore_nccl(executor_manager.param_arrays, executor_manager.grad_arrays, kvstore, executor_manager.param_names)
                    else:
                        _update_params_on_kvstore(executor_manager.param_arrays, executor_manager.grad_arrays, kvstore, executor_manager.param_names)
                else:
                    _update_params(executor_manager.param_arrays, executor_manager.grad_arrays, updater=updater, num_device=len(ctx), kvstore=kvstore, param_names=executor_manager.param_names)
                if monitor is not None:
                    monitor.toc_print()
                executor_manager.update_metric(eval_metric, data_batch.label)
                nbatch += 1
                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric, locals=locals())
                    _multiple_callbacks(batch_end_callback, batch_end_params)
                if epoch_size is not None and nbatch >= epoch_size:
                    do_reset = False
                    break
            if do_reset:
                logger.info('Epoch[%d] Resetting Data Iterator', epoch)
                train_data.reset()
            if epoch_size is None or nbatch >= epoch_size:
                break
        toc = time.time()
        logger.info('Epoch[%d] Time cost=%.3f', epoch, toc - tic)
        if epoch_end_callback or epoch + 1 == end_epoch:
            executor_manager.copy_to(arg_params, aux_params)
        _multiple_callbacks(epoch_end_callback, epoch, symbol, arg_params, aux_params)
        if eval_data:
            eval_metric.reset()
            eval_data.reset()
            total_num_batch = 0
            for i, eval_batch in enumerate(eval_data):
                executor_manager.load_data_batch(eval_batch)
                executor_manager.forward(is_train=False)
                executor_manager.update_metric(eval_metric, eval_batch.label)
                if eval_batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=i, eval_metric=eval_metric, locals=locals())
                    _multiple_callbacks(eval_batch_end_callback, batch_end_params)
                total_num_batch += 1
            if eval_end_callback is not None:
                eval_end_params = BatchEndParam(epoch=epoch, nbatch=total_num_batch, eval_metric=eval_metric, locals=locals())
                _multiple_callbacks(eval_end_callback, eval_end_params)
            eval_data.reset()