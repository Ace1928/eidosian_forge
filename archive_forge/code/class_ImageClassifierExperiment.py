from tensorflow.keras import datasets
import autokeras as ak
from benchmark.experiments import experiment
class ImageClassifierExperiment(experiment.Experiment):

    def get_auto_model(self):
        return ak.ImageClassifier(max_trials=10, directory=self.tmp_dir, overwrite=True)