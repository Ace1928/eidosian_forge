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
def _dict_from_keyval(k, v, json_parse=True):
    d = ret = {}
    keys = k.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    if json_parse:
        try:
            v = json.loads(v.strip('"'))
        except ValueError:
            pass
    d[keys[-1]] = v
    return ret