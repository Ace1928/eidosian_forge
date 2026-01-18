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
def _magic_get_config(k, default):
    d = _magic_config
    keys = k.split('.')
    for k in keys[:-1]:
        d = d.get(k, {})
    return d.get(keys[-1], default)