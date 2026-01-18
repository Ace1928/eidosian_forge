from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotCommandStatus(_messages.Message):
    """The internal status of the command result.

  Enums:
    CodeValueValuesEnum: The status code.

  Fields:
    code: The status code.
    message: The error message.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The status code.

    Values:
      OK: The command succeeded.
      INVALID_ARGUMENT: The command input was invalid.
      DEADLINE_EXCEEDED: The command had passed its expiry time while it was
        still running.
      NOT_FOUND: The resources requested by the command were not found.
      PERMISSION_DENIED: The command failed due to permission errors.
      INTERNAL: The command failed because of some invariants expected by the
        underlying system have been broken. This usually indicates a bug wit
        the system.
      ABORTED: The command was aborted.
      FAILED_PRECONDITION: The command failed because the system is not in a
        state required for the command, e.g. the command inputs cannot be
        found on the server.
      CLEANUP_ERROR: The bot failed to do the cleanup, e.g. unable to delete
        the command working directory or the command process.
      DOWNLOAD_INPUTS_ERROR: The bot failed to download the inputs.
      UNKNOWN: Unknown error.
      UPLOAD_OUTPUTS_ERROR: The bot failed to upload the outputs.
      UPLOAD_OUTPUTS_BYTES_LIMIT_EXCEEDED: The bot tried to upload files
        having a total size that is too large.
      DOCKER_LOGIN_ERROR: The bot failed to login to docker.
      DOCKER_IMAGE_PULL_ERROR: The bot failed to pull docker image.
      DOCKER_IMAGE_EXIST_ERROR: The bot failed to check docker images.
      DUPLICATE_INPUTS: The inputs contain duplicate files.
      DOCKER_IMAGE_PERMISSION_DENIED: The bot doesn't have the permissions to
        pull docker images.
      DOCKER_IMAGE_NOT_FOUND: The docker image cannot be found.
      WORKING_DIR_NOT_FOUND: Working directory is not found.
      WORKING_DIR_NOT_IN_BASE_DIR: Working directory is not under the base
        directory
      DOCKER_UNAVAILABLE: There are issues with docker service/runtime.
      NO_CUDA_CAPABLE_DEVICE: The command failed with "no cuda-capable device
        is detected" error.
      REMOTE_CAS_DOWNLOAD_ERROR: The bot encountered errors from remote CAS
        when downloading blobs.
      REMOTE_CAS_UPLOAD_ERROR: The bot encountered errors from remote CAS when
        uploading blobs.
      LOCAL_CASPROXY_NOT_RUNNING: The local casproxy is not running.
      DOCKER_CREATE_CONTAINER_ERROR: The bot couldn't start the container.
      DOCKER_INVALID_ULIMIT: The docker ulimit is not valid.
      DOCKER_UNKNOWN_RUNTIME: The docker runtime is unknown.
      DOCKER_UNKNOWN_CAPABILITY: The docker capability is unknown.
      DOCKER_UNKNOWN_ERROR: The command failed with unknown docker errors.
      DOCKER_CREATE_COMPUTE_SYSTEM_ERROR: Docker failed to run containers with
        CreateComputeSystem error.
      DOCKER_PREPARELAYER_ERROR: Docker failed to run containers with
        hcsshim::PrepareLayer error.
      DOCKER_INCOMPATIBLE_OS_ERROR: Docker incompatible operating system
        error.
      DOCKER_CREATE_RUNTIME_FILE_NOT_FOUND: Docker failed to create OCI
        runtime because of file not found.
      DOCKER_CREATE_RUNTIME_PERMISSION_DENIED: Docker failed to create OCI
        runtime because of permission denied.
      DOCKER_CREATE_PROCESS_FILE_NOT_FOUND: Docker failed to create process
        because of file not found.
      DOCKER_CREATE_COMPUTE_SYSTEM_INCORRECT_PARAMETER_ERROR: Docker failed to
        run containers with CreateComputeSystem error that involves an
        incorrect parameter (more specific version of
        DOCKER_CREATE_COMPUTE_SYSTEM_ERROR that is user-caused).
      DOCKER_TOO_MANY_SYMBOLIC_LINK_LEVELS: Docker failed to create an overlay
        mount because of too many levels of symbolic links.
      LOCAL_CONTAINER_MANAGER_NOT_RUNNING: The local Container Manager is not
        running.
      DOCKER_IMAGE_VPCSC_PERMISSION_DENIED: Docker failed because a request
        was denied by the organization's policy.
      WORKING_DIR_NOT_RELATIVE: Working directory is not relative
      DOCKER_MISSING_CONTAINER: Docker cannot find the container specified in
        the command. This error is likely to only occur if an asynchronous
        container is not running when the command is run.
      DOCKER_MISSING_BLOB_IN_IMAGE: Docker cannot pull an image because a blob
        is missing in the repo. May be due to a bad/incomplete image push or
        partial deletion of underlying blob layers.
      DOCKER_INVALID_VOLUME: The docker volume specification is invalid (e.g.
        root).
      DOCKER_CREATE_RUNTIME_CANNOT_MOUNT_TO_PROC: Docker failed to create OCI
        runtime because input root cannot be proc.
      DOCKER_START_RUNTIME_FILE_NOT_FOUND: Docker failed to start OCI runtime
        because of file not found.
      DOCKER_CREATE_INVALID_LAYERCHAIN_JSON: Docker failed to run because the
        layerchain json was invalid (see b/234782336).
      INCOMPATIBLE_CUDA_VERSION: Docker failed to create OCI runtime because
        of incompatible cuda version.
      LOCAL_WORKER_MANAGER_NOT_RUNNING: The local Worker Manager is not
        running.
      DOCKER_START_RUNTIME_FILE_FORMAT_ERROR: Docker failed to start OCI
        runtime because of file format error.
      DOCKER_START_RUNTIME_PERMISSION_DENIED: Docker failed to start OCI
        runtime because of permission denied.
      DOCKER_PERMISSION_DENIED: Docker failed because of permission denied.
    """
        OK = 0
        INVALID_ARGUMENT = 1
        DEADLINE_EXCEEDED = 2
        NOT_FOUND = 3
        PERMISSION_DENIED = 4
        INTERNAL = 5
        ABORTED = 6
        FAILED_PRECONDITION = 7
        CLEANUP_ERROR = 8
        DOWNLOAD_INPUTS_ERROR = 9
        UNKNOWN = 10
        UPLOAD_OUTPUTS_ERROR = 11
        UPLOAD_OUTPUTS_BYTES_LIMIT_EXCEEDED = 12
        DOCKER_LOGIN_ERROR = 13
        DOCKER_IMAGE_PULL_ERROR = 14
        DOCKER_IMAGE_EXIST_ERROR = 15
        DUPLICATE_INPUTS = 16
        DOCKER_IMAGE_PERMISSION_DENIED = 17
        DOCKER_IMAGE_NOT_FOUND = 18
        WORKING_DIR_NOT_FOUND = 19
        WORKING_DIR_NOT_IN_BASE_DIR = 20
        DOCKER_UNAVAILABLE = 21
        NO_CUDA_CAPABLE_DEVICE = 22
        REMOTE_CAS_DOWNLOAD_ERROR = 23
        REMOTE_CAS_UPLOAD_ERROR = 24
        LOCAL_CASPROXY_NOT_RUNNING = 25
        DOCKER_CREATE_CONTAINER_ERROR = 26
        DOCKER_INVALID_ULIMIT = 27
        DOCKER_UNKNOWN_RUNTIME = 28
        DOCKER_UNKNOWN_CAPABILITY = 29
        DOCKER_UNKNOWN_ERROR = 30
        DOCKER_CREATE_COMPUTE_SYSTEM_ERROR = 31
        DOCKER_PREPARELAYER_ERROR = 32
        DOCKER_INCOMPATIBLE_OS_ERROR = 33
        DOCKER_CREATE_RUNTIME_FILE_NOT_FOUND = 34
        DOCKER_CREATE_RUNTIME_PERMISSION_DENIED = 35
        DOCKER_CREATE_PROCESS_FILE_NOT_FOUND = 36
        DOCKER_CREATE_COMPUTE_SYSTEM_INCORRECT_PARAMETER_ERROR = 37
        DOCKER_TOO_MANY_SYMBOLIC_LINK_LEVELS = 38
        LOCAL_CONTAINER_MANAGER_NOT_RUNNING = 39
        DOCKER_IMAGE_VPCSC_PERMISSION_DENIED = 40
        WORKING_DIR_NOT_RELATIVE = 41
        DOCKER_MISSING_CONTAINER = 42
        DOCKER_MISSING_BLOB_IN_IMAGE = 43
        DOCKER_INVALID_VOLUME = 44
        DOCKER_CREATE_RUNTIME_CANNOT_MOUNT_TO_PROC = 45
        DOCKER_START_RUNTIME_FILE_NOT_FOUND = 46
        DOCKER_CREATE_INVALID_LAYERCHAIN_JSON = 47
        INCOMPATIBLE_CUDA_VERSION = 48
        LOCAL_WORKER_MANAGER_NOT_RUNNING = 49
        DOCKER_START_RUNTIME_FILE_FORMAT_ERROR = 50
        DOCKER_START_RUNTIME_PERMISSION_DENIED = 51
        DOCKER_PERMISSION_DENIED = 52
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    message = _messages.StringField(2)