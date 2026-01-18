from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
class Constructor(yaml.Constructor):
    """A custom yaml constructor.

    It adds 2 different import capabilities. Assuming __init__.yaml has the
    contents:

    foo:
      a: b
      c: d

    baz:
      - e: f
      - g: h

    The first uses a custom constructor to insert data into your current file,
    so:

    bar: !COMMON foo.a

    results in:

    bar: b

    The second mechanism overrides construct_mapping and construct_sequence to
    post process the data and replace the merge macro with keys from the other
    file. We can't use the custom constructor for this as well because the
    merge key type in yaml is processed before custom constructors which makes
    importing and merging not possible. So:

    bar:
      _COMMON_: foo
      i: j

    results in:

    bar:
      a: b
      c: d
      i: j

    This can also be used to merge list contexts, so:

    bar:
      - _COMMON_baz
      - i: j

    results in:

    bar:
      - e: f
      - g: h
      - i: j

    You may also use the !REF and _REF_ directives in the same way. Instead of
    pulling from the common file, they can pull from an arbitrary yaml file
    somewhere in the googlecloudsdk tree. The syntax looks like:

    bar: !REF googlecloudsdk.foo.bar:a.b.c

    This will load googlecloudsdk/foo/bar.yaml and from that file return the
    a.b.c nested attribute.
    """
    INCLUDE_COMMON_MACRO = '!COMMON'
    MERGE_COMMON_MACRO = '_COMMON_'
    INCLUDE_REF_MACRO = '!REF'
    MERGE_REF_MACRO = '_REF_'

    def construct_mapping(self, *args, **kwargs):
        data = super(Constructor, self).construct_mapping(*args, **kwargs)
        data = self._ConstructMappingHelper(Constructor.MERGE_COMMON_MACRO, self._GetCommonData, data)
        return self._ConstructMappingHelper(Constructor.MERGE_REF_MACRO, self._GetRefData, data)

    def _ConstructMappingHelper(self, macro, source_func, data):
        attribute_path = data.pop(macro, None)
        if not attribute_path:
            return data
        modified_data = {}
        for path in attribute_path.split(','):
            modified_data.update(source_func(path))
        modified_data.update(data)
        return modified_data

    def construct_sequence(self, *args, **kwargs):
        data = super(Constructor, self).construct_sequence(*args, **kwargs)
        data = self._ConstructSequenceHelper(Constructor.MERGE_COMMON_MACRO, self._GetCommonData, data)
        return self._ConstructSequenceHelper(Constructor.MERGE_REF_MACRO, self._GetRefData, data)

    def _ConstructSequenceHelper(self, macro, source_func, data):
        new_list = []
        for i in data:
            if isinstance(i, six.string_types) and i.startswith(macro):
                attribute_path = i[len(macro):]
                for path in attribute_path.split(','):
                    new_list.extend(source_func(path))
            else:
                new_list.append(i)
        return new_list

    def IncludeCommon(self, node):
        attribute_path = self.construct_scalar(node)
        return self._GetCommonData(attribute_path)

    def IncludeRef(self, node):
        attribute_path = self.construct_scalar(node)
        return self._GetRefData(attribute_path)

    def _GetCommonData(self, attribute_path):
        if not common_data:
            raise LayoutException('Command [{}] references [common command] data but it does not exist.'.format(impl_path))
        return self._GetAttribute(common_data, attribute_path, 'common command')

    def _GetRefData(self, path):
        """Loads the YAML data from the given reference.

      A YAML reference must refer to a YAML file and an attribute within that
      file to extract.

      Args:
        path: str, The path of the YAML file to import. It must be in the
          form of: package.module:attribute.attribute, where the module path is
          separated from the sub attributes within the YAML by a ':'.

      Raises:
        LayoutException: If the given module or attribute cannot be loaded.

      Returns:
        The referenced YAML data.
      """
        parts = path.split(':')
        if len(parts) != 2:
            raise LayoutException('Invalid Yaml reference: [{}]. References must be in the format: path(.path)+:attribute(.attribute)*'.format(path))
        path_segments = parts[0].split('.')
        try:
            root_module = importlib.import_module(path_segments[0])
            yaml_path = os.path.join(os.path.dirname(root_module.__file__), *path_segments[1:]) + '.yaml'
            data = _SafeLoadYamlFile(yaml_path)
        except (ImportError, IOError) as e:
            raise LayoutException('Failed to load Yaml reference file [{}]: {}'.format(parts[0], e))
        return self._GetAttribute(data, parts[1], yaml_path)

    def _GetAttribute(self, data, attribute_path, location):
        value = data
        for attribute in attribute_path.split('.'):
            value = value.get(attribute, None)
            if not value:
                raise LayoutException('Command [{}] references [{}] data attribute [{}] in path [{}] but it does not exist.'.format(impl_path, location, attribute, attribute_path))
        return value