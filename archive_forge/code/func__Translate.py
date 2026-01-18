import re
import sys
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