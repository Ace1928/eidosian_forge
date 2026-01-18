import copy
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import EnvConfigDict
Sets missing keys of self to the values given in `defaults`.

        If `defaults` contains keys that already exist in self, don't override
        the values with these defaults.

        Args:
            defaults: The key/value pairs to add to self, but only for those
                keys in `defaults` that don't exist yet in self.

        .. testcode::
            :skipif: True

            from ray.rllib.env.env_context import EnvContext
            env_ctx = EnvContext({"a": 1, "b": 2}, worker_index=0)
            env_ctx.set_defaults({"a": -42, "c": 3})
            print(env_ctx)

        .. testoutput::

            {"a": 1, "b": 2, "c": 3}
        