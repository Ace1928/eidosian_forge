from autokeras.tuners import greedy
class ImageClassifierTuner(greedy.Greedy):

    def __init__(self, **kwargs):
        super().__init__(initial_hps=IMAGE_CLASSIFIER, **kwargs)