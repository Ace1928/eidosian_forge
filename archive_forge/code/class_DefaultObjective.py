from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import metrics_tracking
class DefaultObjective(Objective):
    """Default objective to minimize if not provided by the user."""

    def __init__(self):
        super().__init__(name='default_objective', direction='min')