from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class YamlConfigFile(object):
    """Utility class for searching and editing collections of YamlObjects.

  Attributes:
    item_type: class, YamlConfigObject class type of the items in file
    file_contents: str, YAML contents used to load YamlConfigObjects
    file_path: str, file path that YamlConfigObjects were loaded from
    data: [YamlObject], data loaded from file path. Could be 1 or more objects.
    yaml: str, yaml string representation of object.
  """

    def __init__(self, item_type, file_contents=None, file_path=None):
        self._file_contents = file_contents
        self._file_path = file_path
        self._item_type = item_type
        if not self._file_contents and (not self._file_path):
            raise YamlConfigFileError('Could Not Initialize YamlConfigFile:file_contents And file_path Are Both Empty')
        if self._file_contents:
            try:
                items = yaml.load_all(self._file_contents, round_trip=True)
                self._data = [item_type(x) for x in items]
            except yaml.YAMLParseError as fe:
                raise YamlConfigFileError('Error Parsing Config File: [{}]'.format(fe))
        elif self._file_path:
            try:
                items = yaml.load_all_path(self._file_path, round_trip=True)
                self._data = [item_type(x) for x in items]
            except yaml.FileLoadError as fe:
                raise YamlConfigFileError('Error Loading Config File: [{}]'.format(fe))

    @property
    def item_type(self):
        return self._item_type

    @property
    def data(self):
        return self._data

    @property
    def yaml(self):
        if len(self._data) == 1:
            return str(self._data[0])
        return '---\n'.join([str(x) for x in self._data])

    @property
    def file_contents(self):
        return self._file_contents

    @property
    def file_path(self):
        return self._file_path

    def __str__(self):
        return self.yaml

    def __eq__(self, other):
        if isinstance(other, YamlConfigFile):
            return len(self.data) == len(other.data) and all((x == y for x, y in zip(self.data, other.data)))
        return False

    def FindMatchingItem(self, search_path, value):
        """Find all YamlObjects with matching data at search_path."""
        results = []
        for obj in self.data:
            if obj[search_path] == value:
                results.append(obj)
        return results

    def FindMatchingItemData(self, search_path):
        """Find all data in YamlObjects at search_path."""
        results = []
        for obj in self.data:
            value = obj[search_path]
            if value:
                results.append(value)
        return results

    def SetMatchingItemData(self, object_path, object_value, item_path, item_value, persist=True):
        """Find all matching YamlObjects and set values."""
        results = []
        found_items = self.FindMatchingItem(object_path, object_value)
        for ymlconfig in found_items:
            ymlconfig[item_path] = item_value
            results.append(ymlconfig)
        if persist:
            self.WriteToDisk()
        return results

    def WriteToDisk(self):
        """Overwrite Original Yaml File."""
        if not self.file_path:
            raise YamlConfigFileError('Could Not Write To Config File: Path Is Empty')
        out_file_buf = io.BytesIO()
        tmp_yaml_buf = io.TextIOWrapper(out_file_buf, newline='\n', encoding='utf-8')
        yaml.dump_all_round_trip([x.content for x in self.data], stream=tmp_yaml_buf)
        with files.BinaryFileWriter(self.file_path) as f:
            tmp_yaml_buf.seek(0)
            f.write(out_file_buf.getvalue())