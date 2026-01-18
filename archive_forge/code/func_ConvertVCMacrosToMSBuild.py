import re
import sys
def ConvertVCMacrosToMSBuild(s):
    """Convert the MSVS macros found in the string to the MSBuild equivalent.

  This list is probably not exhaustive.  Add as needed.
  """
    if '$' in s:
        replace_map = {'$(ConfigurationName)': '$(Configuration)', '$(InputDir)': '%(RelativeDir)', '$(InputExt)': '%(Extension)', '$(InputFileName)': '%(Filename)%(Extension)', '$(InputName)': '%(Filename)', '$(InputPath)': '%(Identity)', '$(ParentName)': '$(ProjectFileName)', '$(PlatformName)': '$(Platform)', '$(SafeInputName)': '%(Filename)'}
        for old, new in replace_map.items():
            s = s.replace(old, new)
        s = FixVCMacroSlashes(s)
    return s