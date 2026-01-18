import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@magics_class
class TimerMagic(Magics):
    """
    A line magic for measuring the execution time of multiple cells.

    After you start/reset the timer with '%timer start' you may view
    elapsed time with any subsequent calls to %timer.
    """
    start_time = None

    @staticmethod
    def elapsed_time():
        seconds = time.time() - TimerMagic.start_time
        minutes = seconds // 60
        hours = minutes // 60
        return 'Timer elapsed: %02d:%02d:%02d' % (hours, minutes % 60, seconds % 60)

    @classmethod
    def option_completer(cls, k, v):
        return ['start']

    @line_magic
    def timer(self, line=''):
        """
        Timer magic to print initial date/time information and
        subsequent elapsed time intervals.

        To start the timer, run:

        %timer start

        This will print the start date and time.

        Subsequent calls to %timer will print the elapsed time
        relative to the time when %timer start was called. Subsequent
        calls to %timer start may also be used to reset the timer.
        """
        if line.strip() not in ['', 'start']:
            print('Invalid argument to %timer. For more information consult %timer?')
            return
        elif line.strip() == 'start':
            TimerMagic.start_time = time.time()
            timestamp = time.strftime('%Y/%m/%d %H:%M:%S')
            print(f'Timer start: {timestamp}')
            return
        elif self.start_time is None:
            print('Please start timer with %timer start. For more information consult %timer?')
        else:
            print(self.elapsed_time())