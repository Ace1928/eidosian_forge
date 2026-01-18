import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
class FileSummary(Summary):

    def __init__(self, run):
        super().__init__(run)
        self._fname = os.path.join(run.dir, wandb_lib.filenames.SUMMARY_FNAME)
        self.load()

    def load(self):
        try:
            with open(self._fname) as f:
                self._json_dict = json.load(f)
        except (OSError, ValueError):
            self._json_dict = {}

    def _write(self, commit=False):
        with open(self._fname, 'w') as f:
            f.write(util.json_dumps_safer(self._json_dict))
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if self._h5:
            self._h5.close()
            self._h5 = None
        if wandb.run and wandb.run._jupyter_agent:
            wandb.run._jupyter_agent.start()