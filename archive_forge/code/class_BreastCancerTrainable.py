from typing import Dict, Any, Optional, TYPE_CHECKING
import sklearn.datasets
import sklearn.metrics
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.core import Booster
import pickle
import ray
from ray import train, tune
from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Trial
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
class BreastCancerTrainable(Trainable):

    def setup(self, config):
        self.config = config
        self.nthread = config.pop('nthread', 1)
        self.model: xgb.Booster = None
        data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
        self.train_set = xgb.DMatrix(train_x, label=train_y)
        self.test_set = xgb.DMatrix(test_x, label=test_y)

    def step(self):
        current_resources = self.trial_resources
        if isinstance(current_resources, PlacementGroupFactory):
            self.nthread = current_resources.head_cpus
        else:
            self.nthread = current_resources.cpu
        results = {}
        config = self.config.copy()
        config['nthread'] = int(self.nthread)
        self.model = xgb.train(config, self.train_set, evals=[(self.test_set, 'eval')], verbose_eval=False, xgb_model=self.model, evals_result=results, num_boost_round=1)
        print(config, results)
        return {'eval-logloss': results['eval']['logloss'][-1], 'nthread': self.nthread}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'checkpoint')
        with open(path, 'wb') as outputFile:
            pickle.dump((self.config, self.nthread, self.model.save_raw()), outputFile)

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'checkpoint')
        with open(path, 'rb') as inputFile:
            self.config, self.nthread, raw_model = pickle.load(inputFile)
        self.model = Booster()
        self.model.load_model(bytearray(raw_model))
        data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
        self.train_set = xgb.DMatrix(train_x, label=train_y)
        self.test_set = xgb.DMatrix(test_x, label=test_y)