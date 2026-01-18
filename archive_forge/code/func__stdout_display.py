import math
import sys
import time
import param
from IPython.core.display import clear_output
from ..core.util import ProgressIndicator
def _stdout_display(self, percentage, display=True):
    if clear_output:
        clear_output()
    percent_per_char = 100.0 / self.width
    char_count = int(math.floor(percentage / percent_per_char) if percentage < 100.0 else self.width)
    blank_count = self.width - char_count
    prefix = '\n' if len(self.current_progress) > 1 else ''
    self.out = prefix + '{}[{}{}] {:0.1f}%'.format(self.label + ':\n' if self.label else '', self.fill_char * char_count, ' ' * len(self.fill_char) * blank_count, percentage)
    if display:
        sys.stdout.write(''.join([pg.out for pg in self.current_progress]))
        sys.stdout.flush()
        time.sleep(0.0001)