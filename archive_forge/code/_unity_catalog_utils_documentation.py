from typing import List, Optional
from mlflow.entities.model_registry import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository

    Constructs the full name of a registered model using the active catalog and schema in a spark
    session / context.

    Args:
        name: The model name provided by the user.
        spark: The active spark session.
    