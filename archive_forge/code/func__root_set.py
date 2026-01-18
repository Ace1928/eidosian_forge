import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def _root_set(self, path, new_keys_values):
    json_dict = self._json_dict
    for key in path:
        json_dict = json_dict[key]
    for new_key, new_value in new_keys_values:
        json_dict[new_key] = self._encode(new_value, path + (new_key,))