import itertools
import os
from keystoneauth1.loading import _utils
@property
def _all_opts(self):
    return itertools.chain([self], self.deprecated)