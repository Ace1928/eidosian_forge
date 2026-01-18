import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def _load_class_from_transformers_config(model_name_or_path, revision=None):
    """
    This method retrieves the Transformers AutoClass from the transformers config.
    Using the correct AutoClass allows us to leverage Transformers' model loading
    machinery, which is necessary for supporting models using custom code.
    """
    import transformers
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=True)
    class_name = config.architectures[0]
    if hasattr(transformers, class_name):
        cls = getattr(transformers, class_name)
        return (cls, False)
    else:
        auto_classes = [auto_class for auto_class, module in config.auto_map.items() if module.split('.')[-1] == class_name]
        if len(auto_classes) == 0:
            raise MlflowException(f"Couldn't find a loader class for {class_name}")
        auto_class = auto_classes[0]
        cls = getattr(transformers, auto_class)
        return (cls, True)