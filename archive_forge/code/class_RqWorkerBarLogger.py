from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
class RqWorkerBarLogger(RqWorkerProgressLogger, ProgressBarLogger):

    def __init__(self, job, init_state=None, bars=None, ignored_bars=(), logged_bars='all', min_time_interval=0):
        RqWorkerProgressLogger.__init__(self, job)
        ProgressBarLogger.__init__(self, init_state=init_state, bars=bars, ignored_bars=ignored_bars, logged_bars=logged_bars, min_time_interval=min_time_interval)