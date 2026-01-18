from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils
from autokeras.utils import utils
def _analyze_data(self, dataset):
    input_analysers = [node.get_analyser() for node in self.inputs]
    output_analysers = [head.get_analyser() for head in self._heads]
    analysers = input_analysers + output_analysers
    for x, y in dataset:
        x = nest.flatten(x)
        y = nest.flatten(y)
        for item, analyser in zip(x + y, analysers):
            analyser.update(item)
    for analyser in analysers:
        analyser.finalize()
    for hm, analyser in zip(self.inputs + self._heads, analysers):
        hm.config_from_analyser(analyser)