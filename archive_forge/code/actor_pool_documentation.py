from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
Pushes a new actor into the current list of idle actors.

        Examples:
            .. testcode::

                import ray
                from ray.util.actor_pool import ActorPool

                @ray.remote
                class Actor:
                    def double(self, v):
                        return 2 * v

                a1, a2 = Actor.remote(), Actor.remote()
                pool = ActorPool([a1])
                pool.push(a2)
        