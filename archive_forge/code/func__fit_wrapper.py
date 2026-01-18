import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _fit_wrapper(self, fn, generator=None, *args, **kwargs):
    trigger.call('on_fit')
    keras = sys.modules.get('keras', None)
    tfkeras = sys.modules.get('tensorflow.python.keras', None)
    epochs = kwargs.pop('epochs', None)
    batch_size = kwargs.pop('batch_size', None)
    magic_epochs = _magic_get_config('keras.fit.epochs', None)
    if magic_epochs is not None:
        epochs = magic_epochs
    magic_batch_size = _magic_get_config('keras.fit.batch_size', None)
    if magic_batch_size is not None:
        batch_size = magic_batch_size
    callbacks = kwargs.pop('callbacks', [])
    tb_enabled = _magic_get_config('keras.fit.callbacks.tensorboard.enable', None)
    if tb_enabled:
        k = getattr(self, '_keras_or_tfkeras', None)
        if k:
            tb_duplicate = _magic_get_config('keras.fit.callbacks.tensorboard.duplicate', None)
            tb_overwrite = _magic_get_config('keras.fit.callbacks.tensorboard.overwrite', None)
            tb_present = any([isinstance(cb, k.callbacks.TensorBoard) for cb in callbacks])
            if tb_present and tb_overwrite:
                callbacks = [cb for cb in callbacks if not isinstance(cb, k.callbacks.TensorBoard)]
            if tb_overwrite or tb_duplicate or (not tb_present):
                tb_callback_kwargs = {'log_dir': wandb.run.dir}
                cb_args = ('write_graph', 'histogram_freq', 'update_freq', 'write_grads', 'write_images', 'batch_size')
                for cb_arg in cb_args:
                    v = _magic_get_config('keras.fit.callbacks.tensorboard.' + cb_arg, None)
                    if v is not None:
                        tb_callback_kwargs[cb_arg] = v
                tb_callback = k.callbacks.TensorBoard(**tb_callback_kwargs)
                callbacks.append(tb_callback)
    wandb_enabled = _magic_get_config('keras.fit.callbacks.wandb.enable', None)
    if wandb_enabled:
        wandb_duplicate = _magic_get_config('keras.fit.callbacks.wandb.duplicate', None)
        wandb_overwrite = _magic_get_config('keras.fit.callbacks.wandb.overwrite', None)
        wandb_present = any([isinstance(cb, wandb.keras.WandbCallback) for cb in callbacks])
        if wandb_present and wandb_overwrite:
            callbacks = [cb for cb in callbacks if not isinstance(cb, wandb.keras.WandbCallback)]
        if wandb_overwrite or wandb_duplicate or (not wandb_present):
            wandb_callback_kwargs = {}
            log_gradients = _magic_get_config('keras.fit.callbacks.wandb.log_gradients', None)
            if log_gradients and kwargs.get('x') and kwargs.get('y'):
                wandb_callback_kwargs['log_gradients'] = log_gradients
            cb_args = ('predictions', 'log_weights', 'data_type', 'save_model', 'save_weights_only', 'monitor', 'mode', 'verbose', 'input_type', 'output_type', 'log_evaluation', 'labels')
            for cb_arg in cb_args:
                v = _magic_get_config('keras.fit.callbacks.wandb.' + cb_arg, None)
                if v is not None:
                    wandb_callback_kwargs[cb_arg] = v
            wandb_callback = wandb.keras.WandbCallback(**wandb_callback_kwargs)
            callbacks.append(wandb_callback)
    kwargs['callbacks'] = callbacks
    if epochs is not None:
        kwargs['epochs'] = epochs
    if batch_size is not None:
        kwargs['batch_size'] = batch_size
    if generator:
        return fn(generator, *args, **kwargs)
    return fn(*args, **kwargs)