import os
from traceback import format_exception
from typing import Optional, Union
import colorama
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray._raylet import ActorID, TaskID, WorkerID
from ray.core.generated.common_pb2 import (
from ray.util.annotations import DeveloperAPI, PublicAPI
import setproctitle
@PublicAPI
class RayActorError(RayError):
    """Indicates that the actor died unexpectedly before finishing a task.

    This exception could happen either because the actor process dies while
    executing a task, or because a task is submitted to a dead actor.

    If the actor is dead because of an exception thrown in its creation tasks,
    RayActorError will contain the creation_task_error, which is used to
    reconstruct the exception on the caller side.

    Args:
        cause: The cause of the actor error. `RayTaskError` type means
            the actor has died because of an exception within `__init__`.
            `ActorDiedErrorContext` means the actor has died because of
            unexepected system error. None means the cause is not known.
            Theoretically, this should not happen,
            but it is there as a safety check.
    """

    def __init__(self, cause: Union[RayTaskError, ActorDiedErrorContext]=None):
        self._actor_init_failed = False
        self.base_error_msg = 'The actor died unexpectedly before finishing this task.'
        self._preempted = False
        if not cause:
            self.error_msg = self.base_error_msg
        elif isinstance(cause, RayTaskError):
            self._actor_init_failed = True
            self.actor_id = cause._actor_id
            self.error_msg = f'The actor died because of an error raised in its creation task, {cause.__str__()}'
        else:
            assert isinstance(cause, ActorDiedErrorContext)
            error_msg_lines = [self.base_error_msg]
            error_msg_lines.append(f'\tclass_name: {cause.class_name}')
            error_msg_lines.append(f'\tactor_id: {ActorID(cause.actor_id).hex()}')
            if cause.pid != 0:
                error_msg_lines.append(f'\tpid: {cause.pid}')
            if cause.name != '':
                error_msg_lines.append(f'\tname: {cause.name}')
            if cause.ray_namespace != '':
                error_msg_lines.append(f'\tnamespace: {cause.ray_namespace}')
            if cause.node_ip_address != '':
                error_msg_lines.append(f'\tip: {cause.node_ip_address}')
            error_msg_lines.append(cause.error_message)
            if cause.never_started:
                error_msg_lines.append('The actor never ran - it was cancelled before it started running.')
            if cause.preempted:
                self._preempted = True
                error_msg_lines.append("\tThe actor's node was killed by a spot preemption.")
            self.error_msg = '\n'.join(error_msg_lines)
            self.actor_id = ActorID(cause.actor_id).hex()

    @property
    def actor_init_failed(self) -> bool:
        return self._actor_init_failed

    def __str__(self) -> str:
        return self.error_msg

    @property
    def preempted(self) -> bool:
        return self._preempted

    @staticmethod
    def from_task_error(task_error: RayTaskError):
        return RayActorError(task_error)