import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:
        num_steps (`int`): The number of steps during training. column_names (`List[str]`, *optional*):
            The list of column names for the metrics table (will be inferred from the first call to
            [`~utils.notebook.NotebookTrainingTracker.write_line`] if not set).
    """

    def __init__(self, num_steps, column_names=None):
        super().__init__(num_steps)
        self.inner_table = None if column_names is None else [column_names]
        self.child_bar = None

    def display(self):
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.inner_table is not None:
            self.html_code += text_to_html_table(self.inner_table)
        if self.child_bar is not None:
            self.html_code += self.child_bar.html_code
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (`Dict[str, float]`): The values to display.
        """
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            self.inner_table[0] = columns
            if len(self.inner_table) > 1:
                last_values = self.inner_table[-1]
                first_column = self.inner_table[0][0]
                if last_values[0] != values[first_column]:
                    self.inner_table.append([values[c] if c in values else 'No Log' for c in columns])
                else:
                    new_values = values
                    for c in columns:
                        if c not in new_values.keys():
                            new_values[c] = last_values[columns.index(c)]
                    self.inner_table[-1] = [new_values[c] for c in columns]
            else:
                self.inner_table.append([values[c] for c in columns])

    def add_child(self, total, prefix=None, width=300):
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (`int`): The number of iterations for the child progress bar.
            prefix (`str`, *optional*): A prefix to write on the left of the progress bar.
            width (`int`, *optional*, defaults to 300): The width (in pixels) of the progress bar.
        """
        self.child_bar = NotebookProgressBar(total, prefix=prefix, parent=self, width=width)
        return self.child_bar

    def remove_child(self):
        """
        Closes the child progress bar.
        """
        self.child_bar = None
        self.display()