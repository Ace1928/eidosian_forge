import inspect
import json
import os
import warnings
from keras.src import backend
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.models.variable_mapping import map_trackable_variables
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.utils import io_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
def _get_variable_map(self):
    store = {}
    map_trackable_variables(self, store=store, visited_trackables=set())
    return store