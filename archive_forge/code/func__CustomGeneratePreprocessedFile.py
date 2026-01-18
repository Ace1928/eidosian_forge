import re
import sys
def _CustomGeneratePreprocessedFile(tool, msvs_name):

    def _Translate(value, msbuild_settings):
        tool_settings = _GetMSBuildToolSettings(msbuild_settings, tool)
        if value == '0':
            tool_settings['PreprocessToFile'] = 'false'
            tool_settings['PreprocessSuppressLineNumbers'] = 'false'
        elif value == '1':
            tool_settings['PreprocessToFile'] = 'true'
            tool_settings['PreprocessSuppressLineNumbers'] = 'false'
        elif value == '2':
            tool_settings['PreprocessToFile'] = 'true'
            tool_settings['PreprocessSuppressLineNumbers'] = 'true'
        else:
            raise ValueError('value must be one of [0, 1, 2]; got %s' % value)
    msvs_validator = _Enumeration(['a', 'b', 'c']).ValidateMSVS
    _msvs_validators[tool.msvs_name][msvs_name] = msvs_validator
    msbuild_validator = _boolean.ValidateMSBuild
    msbuild_tool_validators = _msbuild_validators[tool.msbuild_name]
    msbuild_tool_validators['PreprocessToFile'] = msbuild_validator
    msbuild_tool_validators['PreprocessSuppressLineNumbers'] = msbuild_validator
    _msvs_to_msbuild_converters[tool.msvs_name][msvs_name] = _Translate