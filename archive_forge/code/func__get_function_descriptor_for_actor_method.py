from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
def _get_function_descriptor_for_actor_method(language: str, actor_creation_function_descriptor, method_name: str, signature: str):
    """Get function descriptor for cross language actor method call.

    Args:
        language: Target language.
        actor_creation_function_descriptor:
            The function signature for actor creation.
        method_name: The name of actor method.
        signature: The signature for the actor method. When calling Java from Python,
            it should be string in the form of "{length_of_args}".

    Returns:
        Function descriptor for cross language actor method call.
    """
    if language == Language.JAVA:
        return JavaFunctionDescriptor(actor_creation_function_descriptor.class_name, method_name, signature)
    elif language == Language.CPP:
        return CppFunctionDescriptor(method_name, 'PYTHON', actor_creation_function_descriptor.class_name)
    else:
        raise NotImplementedError(f'Cross language remote actor method not support language {language}')