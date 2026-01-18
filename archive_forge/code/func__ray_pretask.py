import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def _ray_pretask(self, key, object_refs):
    self.pb.task_scheduled.remote(key, datetime.now())