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
class SerializationContext:
    """Initialize the serialization library.

    This defines a custom serializer for object refs and also tells ray to
    serialize several exception classes that we define for error handling.
    """

    def __init__(self, worker):
        self.worker = worker
        self._thread_local = threading.local()

        def actor_handle_reducer(obj):
            ray._private.worker.global_worker.check_connected()
            serialized, actor_handle_id = obj._serialization_helper()
            self.add_contained_object_ref(actor_handle_id)
            return (_actor_handle_deserializer, (serialized,))
        self._register_cloudpickle_reducer(ray.actor.ActorHandle, actor_handle_reducer)

        def object_ref_reducer(obj):
            worker = ray._private.worker.global_worker
            worker.check_connected()
            self.add_contained_object_ref(obj)
            obj, owner_address, object_status = worker.core_worker.serialize_object_ref(obj)
            return (_object_ref_deserializer, (obj.binary(), obj.call_site(), owner_address, object_status))
        self._register_cloudpickle_reducer(ray.ObjectRef, object_ref_reducer)

        def object_ref_generator_reducer(obj):
            return (DynamicObjectRefGenerator, (obj._refs,))
        self._register_cloudpickle_reducer(DynamicObjectRefGenerator, object_ref_generator_reducer)
        serialization_addons.apply(self)

    def _register_cloudpickle_reducer(self, cls, reducer):
        pickle.CloudPickler.dispatch[cls] = reducer

    def _unregister_cloudpickle_reducer(self, cls):
        pickle.CloudPickler.dispatch.pop(cls, None)

    def _register_cloudpickle_serializer(self, cls, custom_serializer, custom_deserializer):

        def _CloudPicklerReducer(obj):
            return (custom_deserializer, (custom_serializer(obj),))
        pickle.CloudPickler.dispatch[cls] = _CloudPicklerReducer

    def is_in_band_serialization(self):
        return getattr(self._thread_local, 'in_band', False)

    def set_in_band_serialization(self):
        self._thread_local.in_band = True

    def set_out_of_band_serialization(self):
        self._thread_local.in_band = False

    def get_outer_object_ref(self):
        stack = getattr(self._thread_local, 'object_ref_stack', [])
        return stack[-1] if stack else None

    def get_and_clear_contained_object_refs(self):
        if not hasattr(self._thread_local, 'object_refs'):
            self._thread_local.object_refs = set()
            return set()
        object_refs = self._thread_local.object_refs
        self._thread_local.object_refs = set()
        return object_refs

    def add_contained_object_ref(self, object_ref):
        if self.is_in_band_serialization():
            if not hasattr(self._thread_local, 'object_refs'):
                self._thread_local.object_refs = set()
            self._thread_local.object_refs.add(object_ref)
        else:
            ray._private.worker.global_worker.core_worker.add_object_ref_reference(object_ref)

    def _deserialize_pickle5_data(self, data):
        try:
            in_band, buffers = unpack_pickle5_buffers(data)
            if len(buffers) > 0:
                obj = pickle.loads(in_band, buffers=buffers)
            else:
                obj = pickle.loads(in_band)
        except pickle.pickle.PicklingError:
            raise DeserializationError()
        return obj

    def _deserialize_msgpack_data(self, data, metadata_fields):
        msgpack_data, pickle5_data = split_buffer(data)
        if metadata_fields[0] == ray_constants.OBJECT_METADATA_TYPE_PYTHON:
            python_objects = self._deserialize_pickle5_data(pickle5_data)
        else:
            python_objects = []
        try:

            def _python_deserializer(index):
                return python_objects[index]
            obj = MessagePackSerializer.loads(msgpack_data, _python_deserializer)
        except Exception:
            raise DeserializationError()
        return obj

    def _deserialize_error_info(self, data, metadata_fields):
        assert data
        pb_bytes = self._deserialize_msgpack_data(data, metadata_fields)
        assert pb_bytes
        ray_error_info = RayErrorInfo()
        ray_error_info.ParseFromString(pb_bytes)
        return ray_error_info

    def _deserialize_actor_died_error(self, data, metadata_fields):
        if not data:
            return RayActorError()
        ray_error_info = self._deserialize_error_info(data, metadata_fields)
        assert ray_error_info.HasField('actor_died_error')
        if ray_error_info.actor_died_error.HasField('creation_task_failure_context'):
            return RayError.from_ray_exception(ray_error_info.actor_died_error.creation_task_failure_context)
        else:
            assert ray_error_info.actor_died_error.HasField('actor_died_error_context')
            return RayActorError(cause=ray_error_info.actor_died_error.actor_died_error_context)

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

    def deserialize_objects(self, data_metadata_pairs, object_refs):
        assert len(data_metadata_pairs) == len(object_refs)
        if not hasattr(self._thread_local, 'object_ref_stack'):
            self._thread_local.object_ref_stack = []
        results = []
        for object_ref, (data, metadata) in zip(object_refs, data_metadata_pairs):
            try:
                self._thread_local.object_ref_stack.append(object_ref)
                obj = self._deserialize_object(data, metadata, object_ref)
            except Exception as e:
                logger.exception(e)
                obj = RaySystemError(e, traceback.format_exc())
            finally:
                if self._thread_local.object_ref_stack:
                    self._thread_local.object_ref_stack.pop()
            results.append(obj)
        return results

    def _serialize_to_pickle5(self, metadata, value):
        writer = Pickle5Writer()
        try:
            self.set_in_band_serialization()
            inband = pickle.dumps(value, protocol=5, buffer_callback=writer.buffer_callback)
        except Exception as e:
            self.get_and_clear_contained_object_refs()
            raise e
        finally:
            self.set_out_of_band_serialization()
        return Pickle5SerializedObject(metadata, inband, writer, self.get_and_clear_contained_object_refs())

    def _serialize_to_msgpack(self, value):
        contained_object_refs = []
        if isinstance(value, RayTaskError):
            if issubclass(value.cause.__class__, TaskCancelledError):
                metadata = str(ErrorType.Value('TASK_CANCELLED')).encode('ascii')
                value = value.to_bytes()
            else:
                metadata = str(ErrorType.Value('TASK_EXECUTION_EXCEPTION')).encode('ascii')
                value = value.to_bytes()
        elif isinstance(value, ray.actor.ActorHandle):
            serialized, actor_handle_id = value._serialization_helper()
            contained_object_refs.append(actor_handle_id)
            metadata = ray_constants.OBJECT_METADATA_TYPE_ACTOR_HANDLE
            value = serialized
        else:
            metadata = ray_constants.OBJECT_METADATA_TYPE_CROSS_LANGUAGE
        python_objects = []

        def _python_serializer(o):
            index = len(python_objects)
            python_objects.append(o)
            return index
        msgpack_data = MessagePackSerializer.dumps(value, _python_serializer)
        if python_objects:
            metadata = ray_constants.OBJECT_METADATA_TYPE_PYTHON
            pickle5_serialized_object = self._serialize_to_pickle5(metadata, python_objects)
        else:
            pickle5_serialized_object = None
        return MessagePackSerializedObject(metadata, msgpack_data, contained_object_refs, pickle5_serialized_object)

    def serialize(self, value):
        """Serialize an object.

        Args:
            value: The value to serialize.
        """
        if isinstance(value, bytes):
            return RawSerializedObject(value)
        else:
            return self._serialize_to_msgpack(value)