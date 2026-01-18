import warnings
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.utils.import_hooks import register_post_import_hook

        Get an evaluator instance from the registry based on the name of evaluator
        