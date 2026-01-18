import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
class NotebookProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation, optimized for Jupyter Notebooks or
    Google colab.
    """

    def __init__(self):
        self.training_tracker = None
        self.prediction_bar = None
        self._force_next_update = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.first_column = 'Epoch' if args.evaluation_strategy == IntervalStrategy.EPOCH else 'Step'
        self.training_loss = 0
        self.last_log = 0
        column_names = [self.first_column] + ['Training Loss']
        if args.evaluation_strategy != IntervalStrategy.NO:
            column_names.append('Validation Loss')
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

    def on_step_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if int(state.epoch) == state.epoch else f'{state.epoch:.2f}'
        self.training_tracker.update(state.global_step + 1, comment=f'Epoch {epoch}/{state.num_train_epochs}', force_update=self._force_next_update)
        self._force_next_update = False

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not has_length(eval_dataloader):
            return
        if self.prediction_bar is None:
            if self.training_tracker is not None:
                self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
            else:
                self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
            self.prediction_bar.update(1)
        else:
            self.prediction_bar.update(self.prediction_bar.value + 1)

    def on_predict(self, args, state, control, **kwargs):
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.evaluation_strategy == IntervalStrategy.NO and 'loss' in logs:
            values = {'Training Loss': logs['loss']}
            values['Step'] = state.global_step
            self.training_tracker.write_line(values)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_tracker is not None:
            values = {'Training Loss': 'No log', 'Validation Loss': 'No log'}
            for log in reversed(state.log_history):
                if 'loss' in log:
                    values['Training Loss'] = log['loss']
                    break
            if self.first_column == 'Epoch':
                values['Epoch'] = int(state.epoch)
            else:
                values['Step'] = state.global_step
            metric_key_prefix = 'eval'
            for k in metrics:
                if k.endswith('_loss'):
                    metric_key_prefix = re.sub('\\_loss$', '', k)
            _ = metrics.pop('total_flos', None)
            _ = metrics.pop('epoch', None)
            _ = metrics.pop(f'{metric_key_prefix}_runtime', None)
            _ = metrics.pop(f'{metric_key_prefix}_samples_per_second', None)
            _ = metrics.pop(f'{metric_key_prefix}_steps_per_second', None)
            _ = metrics.pop(f'{metric_key_prefix}_jit_compilation_time', None)
            for k, v in metrics.items():
                splits = k.split('_')
                name = ' '.join([part.capitalize() for part in splits[1:]])
                if name == 'Loss':
                    name = 'Validation Loss'
                values[name] = v
            self.training_tracker.write_line(values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            self._force_next_update = True

    def on_train_end(self, args, state, control, **kwargs):
        self.training_tracker.update(state.global_step, comment=f'Epoch {int(state.epoch)}/{state.num_train_epochs}', force_update=True)
        self.training_tracker = None