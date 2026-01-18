import warnings
from typing import Dict
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.dbfs_artifact_repo import dbfs_artifact_repo_factory
from mlflow.store.artifact.ftp_artifact_repo import FTPArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.r2_artifact_repo import R2ArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository
from mlflow.utils.uri import get_uri_scheme

        Get all registered artifact repositories.

        Returns:
            A dictionary mapping string artifact URI schemes to artifact repositories.
        