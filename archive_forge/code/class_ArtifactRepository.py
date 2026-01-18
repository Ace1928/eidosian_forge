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
@developer_stable
class ArtifactRepository:
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """
    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri
        self.thread_pool = self._create_thread_pool()

    def _create_thread_pool(self):
        return ThreadPoolExecutor(max_workers=self.max_workers)

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        Args:
            local_file: Path to artifact to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
        """
        pass

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        Args:
            local_dir: Directory of local artifacts to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifacts.
        """
        pass

    @abstractmethod
    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        Args:
            path: Relative source path that contains desired artifacts.

        Returns:
            List of artifacts as FileInfo listed directly under path.
        """
        pass

    def _is_directory(self, artifact_path):
        listing = self.list_artifacts(artifact_path)
        return len(listing) > 0

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        """
        Creates a local filesystem location to be used as a destination for downloading the artifact
        specified by `src_artifact_path`. The destination location is a subdirectory of the
        specified `dst_local_dir_path`, which is determined according to the structure of
        `src_artifact_path`. For example, if `src_artifact_path` is `dir1/file1.txt`, then the
        resulting destination path is `<dst_local_dir_path>/dir1/file1.txt`. Local directories are
        created for the resulting destination location if they do not exist.

        Args:
            src_artifact_path: A relative, POSIX-style path referring to an artifact stored
                within the repository's artifact root location. `src_artifact_path` should be
                specified relative to the repository's artifact root location.
            dst_local_dir_path: The absolute path to a local filesystem directory in which the
                local destination path will be contained. The local destination path may be
                contained in a subdirectory of `dst_root_dir` if `src_artifact_path` contains
                subdirectories.

        Returns:
            The absolute path to a local filesystem location to be used as a destination
            for downloading the artifact specified by `src_artifact_path`.
        """
        src_artifact_path = src_artifact_path.rstrip('/')
        dirpath = posixpath.dirname(src_artifact_path)
        local_dir_path = os.path.join(dst_local_dir_path, dirpath)
        local_file_path = os.path.join(dst_local_dir_path, src_artifact_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        return local_file_path

    def _iter_artifacts_recursive(self, path):
        dir_content = [file_info for file_info in self.list_artifacts(path) if file_info.path not in ['.', path]]
        if not dir_content:
            yield FileInfo(path=path, is_dir=True, file_size=None)
            return
        for file_info in dir_content:
            if file_info.is_dir:
                yield from self._iter_artifacts_recursive(file_info.path)
            else:
                yield file_info

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        if dst_path:
            dst_path = os.path.abspath(dst_path)
            if not os.path.exists(dst_path):
                raise MlflowException(message=f'The destination path for downloaded artifacts does not exist! Destination path: {dst_path}', error_code=RESOURCE_DOES_NOT_EXIST)
            elif not os.path.isdir(dst_path):
                raise MlflowException(message=f'The destination path for downloaded artifacts must be a directory! Destination path: {dst_path}', error_code=INVALID_PARAMETER_VALUE)
        else:
            dst_path = create_tmp_dir()

        def _download_file(src_artifact_path, dst_local_dir_path):
            dst_local_file_path = self._create_download_destination(src_artifact_path=src_artifact_path, dst_local_dir_path=dst_local_dir_path)
            return self.thread_pool.submit(self._download_file, remote_file_path=src_artifact_path, local_path=dst_local_file_path)
        futures = {}
        if self._is_directory(artifact_path):
            for file_info in self._iter_artifacts_recursive(artifact_path):
                if file_info.is_dir:
                    os.makedirs(os.path.join(dst_path, file_info.path), exist_ok=True)
                else:
                    fut = _download_file(file_info.path, dst_path)
                    futures[fut] = file_info.path
        else:
            fut = _download_file(artifact_path, dst_path)
            futures[fut] = artifact_path
        failed_downloads = {}
        with ArtifactProgressBar.files(desc='Downloading artifacts', total=len(futures)) as pbar:
            if len(futures) >= 10 and pbar.pbar:
                _logger.info(f'The progress bar can be disabled by setting the environment variable {MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR} to false')
            for f in as_completed(futures):
                try:
                    f.result()
                    pbar.update()
                except Exception as e:
                    path = futures[f]
                    failed_downloads[path] = e
        if failed_downloads:
            template = '##### File {path} #####\n{error}'
            failures = '\n'.join((template.format(path=path, error=error) for path, error in failed_downloads.items()))
            raise MlflowException(message=f'The following failures occurred while downloading one or more artifacts from {self.artifact_uri}:\n{_truncate_error(failures)}')
        return os.path.join(dst_path, artifact_path)

    @abstractmethod
    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        Args:
            remote_file_path: Source path to the remote file, relative to the root
                directory of the artifact repository.
            local_path: The path to which to save the downloaded file.
        """
        pass

    def delete_artifacts(self, artifact_path=None):
        """
        Delete the artifacts at the specified location.
        Supports the deletion of a single file or of a directory. Deletion of a directory
        is recursive.

        Args:
            artifact_path: Path of the artifact to delete.
        """
        pass

    @property
    def max_workers(self) -> int:
        """Compute the number of workers to use for multi-threading."""
        num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
        return min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)