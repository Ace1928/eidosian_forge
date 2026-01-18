import re
from googlecloudsdk.command_lib.run import exceptions
def _ExtractRuntimeVersionFromBaseImage(base_image: str) -> str:
    match = re.search(RUNTIME_FROM_BASE_IMAGE_PATTERN, base_image)
    return match.group(1) if match else None