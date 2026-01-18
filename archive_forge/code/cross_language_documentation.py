from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
Get function descriptor for cross language actor method call.

    Args:
        language: Target language.
        actor_creation_function_descriptor:
            The function signature for actor creation.
        method_name: The name of actor method.
        signature: The signature for the actor method. When calling Java from Python,
            it should be string in the form of "{length_of_args}".

    Returns:
        Function descriptor for cross language actor method call.
    