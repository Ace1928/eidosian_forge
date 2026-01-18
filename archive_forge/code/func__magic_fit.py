import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _magic_fit(self, x=None, y=None, batch_size=None, epochs=1, *args, **kwargs):
    if hasattr(self, '_wandb_internal_model'):
        return self._fit(*args, x=x, y=y, batch_size=batch_size, epochs=epochs, **kwargs)
    return _fit_wrapper(self, self._fit, *args, x=x, y=y, batch_size=batch_size, epochs=epochs, **kwargs)