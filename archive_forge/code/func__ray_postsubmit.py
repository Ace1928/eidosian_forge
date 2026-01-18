import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def _ray_postsubmit(self, task, key, deps, object_ref):
    self.pb.submit.remote(key, deps, datetime.now())