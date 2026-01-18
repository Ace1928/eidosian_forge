from contextlib import contextmanager
import os
import shutil
import sys
import time
import logging
import inspect
import pprint
import subprocess
import textwrap
class Timed:
    """Utility function for timing portions of your python script

    Parameters
    ----------
    msg : str
    timer : callable
        You can switch to e.g. ``time.process_time`` but note that if other
        programs are called from e.g. ``subprocess.Popen`` the time spent
        in those subprocesses will not be included.

    Examples
    --------
    >>> t = Timed("Counting stars...").tic(); stars.count(); t.toc_and_print()  # doctest: +SKIP
    Counting stars...                                          (42.0 s) [  ok]
    >>> with Timed("Counting sheep..."):  # doctest: +SKIP
    ...     n_sheep = animals.count('sheep')
    ...
    Counting sheep...                                          (17.2 s) [  ok]

    """
    counting = False

    def __init__(self, msg=None, timer=time.time, fmt_s='.1f', out=sys.stdout):
        self.msg = msg
        self.out = out
        self.fmt_s = fmt_s
        sys.stdout.flush()
        self.timer = timer

    def tic(self):
        if self.msg is not None:
            self.out.write(self.msg)
            self.out.flush()
        self.counting = True
        self.t = self.timer()
        return self

    def toc(self, ok=True):
        if self.counting:
            t = self.timer() - self.t
            if self.msg is not None:
                if ok:
                    status = 'ok'
                    color = c.ok
                else:
                    status = 'error'
                    color = c.fail
                self.out.write('%{}s\n'.format(shutil.get_terminal_size()[0] - len(self.msg)) % ('(%{fmt_s} s) [{c}%5s{r}]'.format(fmt_s=self.fmt_s, c=color, r=c.endc) % (t, status)))
                self.out.flush()
            return t
        else:
            raise ValueError('Not counting, did you forget to call ``.tic()`` method?')

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, traceback):
        self.toc(not exc_type)