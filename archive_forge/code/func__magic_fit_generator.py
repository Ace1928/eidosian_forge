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
def _magic_fit_generator(self, generator, steps_per_epoch=None, epochs=1, *args, **kwargs):
    return _fit_wrapper(self, self._fit_generator, *args, generator=generator, steps_per_epoch=steps_per_epoch, epochs=epochs, **kwargs)