import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None
        if not any((isinstance(cb, DefaultFlowCallback) for cb in self.callbacks)):
            logger.warning("The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n" + 'should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of' + 'callbacks is\n:' + self.callback_list)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(f'You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current' + 'list of callbacks is\n:' + self.callback_list)
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return '\n'.join((cb.__class__.__name__ for cb in self.callbacks))

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_init_end', args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event('on_train_begin', args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_train_end', args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event('on_epoch_begin', args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_epoch_end', args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event('on_step_begin', args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_substep_end', args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_step_end', args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event('on_evaluate', args, state, control, metrics=metrics)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        return self.call_event('on_predict', args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event('on_save', args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event('on_log', args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_prediction_step', args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(args, state, control, model=self.model, tokenizer=self.tokenizer, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, train_dataloader=self.train_dataloader, eval_dataloader=self.eval_dataloader, **kwargs)
            if result is not None:
                control = result
        return control