from keras_tuner.src import errors
from keras_tuner.src.api_export import keras_tuner_export
class DefaultHyperModel(HyperModel):
    """Produces HyperModel from a model building function."""

    def __init__(self, build, name=None, tunable=True):
        super().__init__(name=name)
        self.build = build