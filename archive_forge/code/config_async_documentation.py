from google.api_core import retry_async
from google.api_core.gapic_v1 import config
from google.api_core.gapic_v1.config import MethodConfig  # noqa: F401
Creates default retry and timeout objects for each method in a gapic
    interface config with AsyncIO semantics.

    Args:
        interface_config (Mapping): The interface config section of the full
            gapic library config. For example, If the full configuration has
            an interface named ``google.example.v1.ExampleService`` you would
            pass in just that interface's configuration, for example
            ``gapic_config['interfaces']['google.example.v1.ExampleService']``.

    Returns:
        Mapping[str, MethodConfig]: A mapping of RPC method names to their
            configuration.
    