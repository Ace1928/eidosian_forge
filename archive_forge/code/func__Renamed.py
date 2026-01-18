import re
import sys
def _Renamed(tool, msvs_name, msbuild_name, setting_type):
    """Defines a setting for which the name has changed.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    msvs_name: the name of the MSVS setting.
    msbuild_name: the name of the MSBuild setting.
    setting_type: the type of this setting.
  """

    def _Translate(value, msbuild_settings):
        msbuild_tool_settings = _GetMSBuildToolSettings(msbuild_settings, tool)
        msbuild_tool_settings[msbuild_name] = setting_type.ConvertToMSBuild(value)
    _msvs_validators[tool.msvs_name][msvs_name] = setting_type.ValidateMSVS
    _msbuild_validators[tool.msbuild_name][msbuild_name] = setting_type.ValidateMSBuild
    _msvs_to_msbuild_converters[tool.msvs_name][msvs_name] = _Translate