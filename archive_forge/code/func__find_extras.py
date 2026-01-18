import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
def _find_extras(self, service_name, type_name, api_version):
    """Creates an iterator over all the extras data."""
    for extras_type in self.extras_types:
        extras_name = f'{type_name}.{extras_type}-extras'
        full_path = os.path.join(service_name, api_version, extras_name)
        try:
            yield self.load_data(full_path)
        except DataNotFoundError:
            pass