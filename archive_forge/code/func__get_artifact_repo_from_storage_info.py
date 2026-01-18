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
def _get_artifact_repo_from_storage_info(storage_location: str, scoped_token: TemporaryCredentials) -> ArtifactRepository:
    credential_type = scoped_token.WhichOneof('credentials')
    if credential_type == 'aws_temp_credentials':
        import boto3
        from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
        aws_creds = scoped_token.aws_temp_credentials
        return OptimizedS3ArtifactRepository(artifact_uri=storage_location, access_key_id=aws_creds.access_key_id, secret_access_key=aws_creds.secret_access_key, session_token=aws_creds.session_token)
    elif credential_type == 'azure_user_delegation_sas':
        from azure.core.credentials import AzureSasCredential
        from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
        sas_token = scoped_token.azure_user_delegation_sas.sas_token
        return AzureDataLakeArtifactRepository(artifact_uri=storage_location, credential=AzureSasCredential(sas_token))
    elif credential_type == 'gcp_oauth_token':
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials
        from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
        credentials = Credentials(scoped_token.gcp_oauth_token.oauth_token)
        client = Client(project='mlflow', credentials=credentials)
        return GCSArtifactRepository(artifact_uri=storage_location, client=client)
    elif credential_type == 'r2_temp_credentials':
        from mlflow.store.artifact.r2_artifact_repo import R2ArtifactRepository
        r2_creds = scoped_token.r2_temp_credentials
        return R2ArtifactRepository(artifact_uri=storage_location, access_key_id=r2_creds.access_key_id, secret_access_key=r2_creds.secret_access_key, session_token=r2_creds.session_token)
    else:
        raise MlflowException(f'Got unexpected credential type {credential_type} when attempting to access model version files in Unity Catalog. Try upgrading to the latest version of the MLflow Python client.')