import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def _deserialize_object(self, data, metadata, object_ref):
    if metadata:
        metadata_fields = metadata.split(b',')
        if metadata_fields[0] in [ray_constants.OBJECT_METADATA_TYPE_CROSS_LANGUAGE, ray_constants.OBJECT_METADATA_TYPE_PYTHON]:
            return self._deserialize_msgpack_data(data, metadata_fields)
        if metadata_fields[0] == ray_constants.OBJECT_METADATA_TYPE_RAW:
            if data is None:
                return b''
            return data.to_pybytes()
        elif metadata_fields[0] == ray_constants.OBJECT_METADATA_TYPE_ACTOR_HANDLE:
            obj = self._deserialize_msgpack_data(data, metadata_fields)
            return _actor_handle_deserializer(obj)
        try:
            error_type = int(metadata_fields[0])
        except Exception:
            raise Exception(f"Can't deserialize object: {object_ref}, metadata: {metadata}")
        if error_type == ErrorType.Value('TASK_EXECUTION_EXCEPTION'):
            obj = self._deserialize_msgpack_data(data, metadata_fields)
            return RayError.from_bytes(obj)
        elif error_type == ErrorType.Value('WORKER_DIED'):
            return WorkerCrashedError()
        elif error_type == ErrorType.Value('ACTOR_DIED'):
            return self._deserialize_actor_died_error(data, metadata_fields)
        elif error_type == ErrorType.Value('LOCAL_RAYLET_DIED'):
            return LocalRayletDiedError()
        elif error_type == ErrorType.Value('TASK_CANCELLED'):
            try:
                error_message = ''
                if data:
                    error_info = self._deserialize_error_info(data, metadata_fields)
                    error_message = error_info.error_message
                return TaskCancelledError(error_message=error_message)
            except google.protobuf.message.DecodeError:
                obj = self._deserialize_msgpack_data(data, metadata_fields)
                return RayError.from_bytes(obj)
        elif error_type == ErrorType.Value('OBJECT_LOST'):
            return ObjectLostError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OBJECT_FETCH_TIMED_OUT'):
            return ObjectFetchTimedOutError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OUT_OF_DISK_ERROR'):
            return OutOfDiskError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OUT_OF_MEMORY'):
            error_info = self._deserialize_error_info(data, metadata_fields)
            return OutOfMemoryError(error_info.error_message)
        elif error_type == ErrorType.Value('NODE_DIED'):
            error_info = self._deserialize_error_info(data, metadata_fields)
            return NodeDiedError(error_info.error_message)
        elif error_type == ErrorType.Value('OBJECT_DELETED'):
            return ReferenceCountingAssertionError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OBJECT_FREED'):
            return ObjectFreedError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OWNER_DIED'):
            return OwnerDiedError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OBJECT_UNRECONSTRUCTABLE'):
            return ObjectReconstructionFailedError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OBJECT_UNRECONSTRUCTABLE_MAX_ATTEMPTS_EXCEEDED'):
            return ObjectReconstructionFailedMaxAttemptsExceededError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('OBJECT_UNRECONSTRUCTABLE_LINEAGE_EVICTED'):
            return ObjectReconstructionFailedLineageEvictedError(object_ref.hex(), object_ref.owner_address(), object_ref.call_site())
        elif error_type == ErrorType.Value('RUNTIME_ENV_SETUP_FAILED'):
            error_info = self._deserialize_error_info(data, metadata_fields)
            error_msg = ''
            if error_info.HasField('runtime_env_setup_failed_error'):
                error_msg = error_info.runtime_env_setup_failed_error.error_message
            return RuntimeEnvSetupError(error_message=error_msg)
        elif error_type == ErrorType.Value('TASK_PLACEMENT_GROUP_REMOVED'):
            return TaskPlacementGroupRemoved()
        elif error_type == ErrorType.Value('ACTOR_PLACEMENT_GROUP_REMOVED'):
            return ActorPlacementGroupRemoved()
        elif error_type == ErrorType.Value('TASK_UNSCHEDULABLE_ERROR'):
            error_info = self._deserialize_error_info(data, metadata_fields)
            return TaskUnschedulableError(error_info.error_message)
        elif error_type == ErrorType.Value('ACTOR_UNSCHEDULABLE_ERROR'):
            error_info = self._deserialize_error_info(data, metadata_fields)
            return ActorUnschedulableError(error_info.error_message)
        elif error_type == ErrorType.Value('END_OF_STREAMING_GENERATOR'):
            return ObjectRefStreamEndOfStreamError()
        else:
            return RaySystemError('Unrecognized error type ' + str(error_type))
    elif data:
        raise ValueError('non-null object should always have metadata')
    else:
        return PlasmaObjectNotAvailable