import re
import sys
class _Enumeration(_Type):
    """Type of settings that is an enumeration.

  In MSVS, the values are indexes like '0', '1', and '2'.
  MSBuild uses text labels that are more representative, like 'Win32'.

  Constructor args:
    label_list: an array of MSBuild labels that correspond to the MSVS index.
        In the rare cases where MSVS has skipped an index value, None is
        used in the array to indicate the unused spot.
    new: an array of labels that are new to MSBuild.
  """

    def __init__(self, label_list, new=None):
        _Type.__init__(self)
        self._label_list = label_list
        self._msbuild_values = {value for value in label_list if value is not None}
        if new is not None:
            self._msbuild_values.update(new)

    def ValidateMSVS(self, value):
        self.ConvertToMSBuild(value)

    def ValidateMSBuild(self, value):
        if value not in self._msbuild_values:
            raise ValueError('unrecognized enumerated value %s' % value)

    def ConvertToMSBuild(self, value):
        index = int(value)
        if index < 0 or index >= len(self._label_list):
            raise ValueError('index value (%d) not in expected range [0, %d)' % (index, len(self._label_list)))
        label = self._label_list[index]
        if label is None:
            raise ValueError('converted value for %s not specified.' % value)
        return label