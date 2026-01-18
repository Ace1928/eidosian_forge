import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
class NotebookProgressBar:
    """
    A progress par for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (`int`) -- The number of iterations to do at the beginning while ignoring `update_every`.
        - **update_every** (`float`) -- Since calling the time takes some time, we only do it every presumed
          `update_every` seconds. The progress bar uses the average time passed up until now to guess the next value
          for which it will call the update.

    Args:
        total (`int`):
            The total number of iterations to reach.
        prefix (`str`, *optional*):
            A prefix to add before the progress bar.
        leave (`bool`, *optional*, defaults to `True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            [`~utils.notebook.NotebookProgressBar.close`] method to make the bar disappear.
        parent ([`~notebook.NotebookTrainingTracker`], *optional*):
            A parent object (like [`~utils.notebook.NotebookTrainingTracker`]) that spawns progress bars and handle
            their display. If set, the object passed must have a `display()` method.
        width (`int`, *optional*, defaults to 300):
            The width (in pixels) that the bar will take.

    Example:

    ```python
    import time

    pbar = NotebookProgressBar(100)
    for val in range(100):
        pbar.update(val)
        time.sleep(0.07)
    pbar.update(100)
    ```"""
    warmup = 5
    update_every = 0.2

    def __init__(self, total: int, prefix: Optional[str]=None, leave: bool=True, parent: Optional['NotebookTrainingTracker']=None, width: int=300):
        self.total = total
        self.prefix = '' if prefix is None else prefix
        self.leave = leave
        self.parent = parent
        self.width = width
        self.last_value = None
        self.comment = None
        self.output = None

    def update(self, value: int, force_update: bool=False, comment: str=None):
        """
        The main method to update the progress bar to `value`.

        Args:
            value (`int`):
                The value to use. Must be between 0 and `total`.
            force_update (`bool`, *optional*, defaults to `False`):
                Whether or not to force and update of the internal state and display (by default, the bar will wait for
                `value` to reach the value it predicted corresponds to a time of more than the `update_every` attribute
                since the last update to avoid adding boilerplate).
            comment (`str`, *optional*):
                A comment to add on the left of the progress bar.
        """
        self.value = value
        if comment is not None:
            self.comment = comment
        if self.last_value is None:
            self.start_time = self.last_time = time.time()
            self.start_value = self.last_value = value
            self.elapsed_time = self.predicted_remaining = None
            self.first_calls = self.warmup
            self.wait_for = 1
            self.update_bar(value)
        elif value <= self.last_value and (not force_update):
            return
        elif force_update or self.first_calls > 0 or value >= min(self.last_value + self.wait_for, self.total):
            if self.first_calls > 0:
                self.first_calls -= 1
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
            if value > self.start_value:
                self.average_time_per_item = self.elapsed_time / (value - self.start_value)
            else:
                self.average_time_per_item = None
            if value >= self.total:
                value = self.total
                self.predicted_remaining = None
                if not self.leave:
                    self.close()
            elif self.average_time_per_item is not None:
                self.predicted_remaining = self.average_time_per_item * (self.total - value)
            self.update_bar(value)
            self.last_value = value
            self.last_time = current_time
            if self.average_time_per_item is None or self.average_time_per_item == 0:
                self.wait_for = 1
            else:
                self.wait_for = max(int(self.update_every / self.average_time_per_item), 1)

    def update_bar(self, value, comment=None):
        spaced_value = ' ' * (len(str(self.total)) - len(str(value))) + str(value)
        if self.elapsed_time is None:
            self.label = f'[{spaced_value}/{self.total} : < :'
        elif self.predicted_remaining is None:
            self.label = f'[{spaced_value}/{self.total} {format_time(self.elapsed_time)}'
        else:
            self.label = f'[{spaced_value}/{self.total} {format_time(self.elapsed_time)} < {format_time(self.predicted_remaining)}'
            if self.average_time_per_item == 0:
                self.label += ', +inf it/s'
            else:
                self.label += f', {1 / self.average_time_per_item:.2f} it/s'
        self.label += ']' if self.comment is None or len(self.comment) == 0 else f', {self.comment}]'
        self.display()

    def display(self):
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.parent is not None:
            self.parent.display()
            return
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def close(self):
        """Closes the progress bar."""
        if self.parent is None and self.output is not None:
            self.output.update(disp.HTML(''))