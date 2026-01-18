import re
from googlecloudsdk.command_lib.run import exceptions
def _BuildGcrUrl(base_image: str, runtime_version: str) -> str:
    return GCR_BUILDER_URL.format(runtime=_SplitVersionFromRuntime(base_image), builder_version=runtime_version)