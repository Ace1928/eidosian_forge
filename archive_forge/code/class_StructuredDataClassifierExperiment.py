import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import autokeras as ak
from benchmark.experiments import experiment
class StructuredDataClassifierExperiment(experiment.Experiment):

    def get_auto_model(self):
        return ak.StructuredDataClassifier(max_trials=10, directory=self.tmp_dir, overwrite=True)