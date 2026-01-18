import importlib
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, NamedTuple, Optional
import numpy as np
import pandas
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tensorflow.callback import MlflowCallback, MlflowModelCheckpointCallback  # noqa: F401
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.types.schema import TensorSpec
from mlflow.utils import is_iterator
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _load_tf1_estimator_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Load a specified TensorFlow model consisting of a TensorFlow metagraph and signature definition
    from a serialized TensorFlow ``SavedModel`` collection.

    Args:
        tf_saved_model_dir: The local filesystem path or run-relative artifact path to the model.
        tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
            serialized ``SavedModel`` object. For more information, see the
            ``tags`` parameter of the `tf.saved_model.builder.SavedModelBuilder
            method <https://www.tensorflow.org/api_docs/python/tf/saved_model/
            builder/SavedModelBuilder#add_meta_graph>`_.
        tf_signature_def_key: A string identifying the input/output signature associated with the
            model. This is a key within the serialized ``SavedModel``'s
            signature definition mapping. For more information, see the
            ``signature_def_map`` parameter of the
            ``tf.saved_model.builder.SavedModelBuilder`` method.

    Returns:
        A callable graph (tensorflow.function) that takes inputs and returns inferences.
    """
    import tensorflow as tf
    loaded = tf.saved_model.load(tags=tf_meta_graph_tags, export_dir=tf_saved_model_dir)
    loaded_sig = loaded.signatures
    if tf_signature_def_key not in loaded_sig:
        raise MlflowException(f'Could not find signature def key {tf_signature_def_key}. Available keys are: {list(loaded_sig.keys())}')
    return loaded_sig[tf_signature_def_key]