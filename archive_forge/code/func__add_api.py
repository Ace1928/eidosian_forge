import importlib
import os
from .. import ports
def _add_api(self, kwargs):
    if self.api and 'api' not in kwargs:
        kwargs['api'] = self.api
    return kwargs