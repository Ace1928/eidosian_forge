import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def _ray_finish(self, result):
    print('All tasks are completed.')