import re
import sys
def _ConvertedToAdditionalOption(tool, msvs_name, flag):
    """Defines a setting that's handled via a command line option in MSBuild.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    msvs_name: the name of the MSVS setting that if 'true' becomes a flag
    flag: the flag to insert at the end of the AdditionalOptions
  """

    def _Translate(value, msbuild_settings):
        if value == 'true':
            tool_settings = _GetMSBuildToolSettings(msbuild_settings, tool)
            if 'AdditionalOptions' in tool_settings:
                new_flags = '{} {}'.format(tool_settings['AdditionalOptions'], flag)
            else:
                new_flags = flag
            tool_settings['AdditionalOptions'] = new_flags
    _msvs_validators[tool.msvs_name][msvs_name] = _boolean.ValidateMSVS
    _msvs_to_msbuild_converters[tool.msvs_name][msvs_name] = _Translate