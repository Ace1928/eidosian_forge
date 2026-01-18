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
def _build_hyper_pipeline(self, dataset):
    self.tuner.hyper_pipeline = pipeline.HyperPipeline(inputs=[node.get_hyper_preprocessors() for node in self.inputs], outputs=[head.get_hyper_preprocessors() for head in self._heads])
    self.tuner.hypermodel.hyper_pipeline = self.tuner.hyper_pipeline