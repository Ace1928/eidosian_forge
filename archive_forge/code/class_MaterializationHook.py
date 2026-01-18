import asyncio
import os
from types import FunctionType
from typing import Sequence
import ray
from ray.util.client.common import ClientObjectRef
from modin.error_message import ErrorMessage
class MaterializationHook:
    """The Hook is called during the materialization and allows performing pre/post computations."""

    def pre_materialize(self):
        """
        Get an object reference to be materialized or a pre-computed value.

        Returns
        -------
        ray.ObjectRef or object
        """
        raise NotImplementedError()

    def post_materialize(self, materialized):
        """
        Perform computations on the materialized object.

        Parameters
        ----------
        materialized : object
            The materialized object to be post-computed.

        Returns
        -------
        object
            The post-computed object.
        """
        raise NotImplementedError()

    def __reduce__(self):
        """
        Replace this hook with the materialized object on serialization.

        Returns
        -------
        tuple
        """
        data = RayWrapper.materialize(self)
        if not isinstance(data, int):
            raise NotImplementedError('Only integers are currently supported')
        return (int, (data,))