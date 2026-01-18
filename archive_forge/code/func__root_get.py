import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def _root_get(self, path, child_dict):
    json_dict = self._json_dict
    for key in path[:-1]:
        json_dict = json_dict[key]
    key = path[-1]
    if key in json_dict:
        child_dict[key] = self._decode(path, json_dict[key])