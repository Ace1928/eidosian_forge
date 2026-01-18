import re
from googlecloudsdk.command_lib.run import exceptions
def FunctionBuilder(base_image: str) -> str:
    runtime = _ExtractRuntimeVersionFromBaseImage(base_image)
    if runtime in BUILDER_22:
        runtime_version = 22
    elif runtime in BUILDER_18:
        runtime_version = 18
    else:
        raise exceptions.InvalidRuntimeLanguage(base_image)
    return _BuildGcrUrl(runtime, runtime_version)