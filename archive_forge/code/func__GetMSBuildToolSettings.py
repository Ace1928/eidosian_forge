import re
import sys
def _GetMSBuildToolSettings(msbuild_settings, tool):
    """Returns an MSBuild tool dictionary.  Creates it if needed."""
    return msbuild_settings.setdefault(tool.msbuild_name, {})