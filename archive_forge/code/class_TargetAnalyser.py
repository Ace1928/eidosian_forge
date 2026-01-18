import numpy as np
from autokeras.engine import analyser
class TargetAnalyser(analyser.Analyser):

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name