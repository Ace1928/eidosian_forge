import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
def _potential_locations(self, name=None, must_exist=False, is_dir=False):
    for path in self.search_paths:
        if os.path.isdir(path):
            full_path = path
            if name is not None:
                full_path = os.path.join(path, name)
            if not must_exist:
                yield full_path
            elif is_dir and os.path.isdir(full_path):
                yield full_path
            elif os.path.exists(full_path):
                yield full_path