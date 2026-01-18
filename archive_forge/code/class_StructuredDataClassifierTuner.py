from autokeras.tuners import greedy
class StructuredDataClassifierTuner(greedy.Greedy):

    def __init__(self, **kwargs):
        super().__init__(initial_hps=STRUCTURED_DATA_CLASSIFIER, **kwargs)