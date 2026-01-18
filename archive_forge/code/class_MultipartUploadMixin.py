import logging
import os
import posixpath
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from mlflow.entities.file_info import FileInfo
from mlflow.entities.multipart_upload import CreateMultipartUploadResponse, MultipartUploadPart
from mlflow.environment_variables import MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import developer_stable
from mlflow.utils.file_utils import ArtifactProgressBar, create_tmp_dir
from mlflow.utils.validation import bad_path_message, path_not_unique
class MultipartUploadMixin(ABC):

    @abstractmethod
    def create_multipart_upload(self, local_file: str, num_parts: int, artifact_path: Optional[str]=None) -> CreateMultipartUploadResponse:
        """
        Initiate a multipart upload and retrieve the pre-signed upload URLS and upload id.

        Args:
            local_file: Path of artifact to upload.
            num_parts: Number of parts to upload. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """
        pass

    @abstractmethod
    def complete_multipart_upload(self, local_file: str, upload_id: str, parts: List[MultipartUploadPart], artifact_path: Optional[str]=None) -> None:
        """
        Complete a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            parts: A list containing the metadata of each part that has been uploaded.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """
        pass

    @abstractmethod
    def abort_multipart_upload(self, local_file: str, upload_id: str, artifact_path: Optional[str]=None) -> None:
        """
        Abort a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """
        pass