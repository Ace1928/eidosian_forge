from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
class _LvObjectConstructor(yaml.constructor.RoundTripConstructor):
    """Location/value object constructor that works for all types.

  The object has these attributes:
    lc.line: The start line of the value in the input file.
    lc.col: The start column of the value in the input file.
    value: The value.
  """
    _initialized = False

    def __init__(self, *args, **kwargs):
        super(_LvObjectConstructor, self).__init__(*args, **kwargs)
        self._Initialize()

    @classmethod
    def _Initialize(cls):
        if not cls._initialized:
            cls._initialized = True
            cls.add_constructor('tag:yaml.org,2002:null', cls.construct_yaml_null)
            cls.add_constructor('tag:yaml.org,2002:bool', cls.construct_yaml_bool)
            cls.add_constructor('tag:yaml.org,2002:int', cls.construct_yaml_int)
            cls.add_constructor('tag:yaml.org,2002:float', cls.construct_yaml_float)
            cls.add_constructor('tag:yaml.org,2002:map', cls.construct_yaml_map)
            cls.add_constructor('tag:yaml.org,2002:omap', cls.construct_yaml_omap)
            cls.add_constructor('tag:yaml.org,2002:seq', cls.construct_yaml_seq)

    def _ScalarType(self, node):
        if isinstance(node.value, six.string_types):
            if node.style == '|':
                return _LvPreservedScalarString(node.value)
            if self._preserve_quotes:
                if node.style == "'":
                    return _LvSingleQuotedScalarString(node.value)
                if node.style == '"':
                    return _LvDoubleQuotedScalarString(node.value)
        return _LvString(node.value)

    def _ScalarObject(self, node, value, raw=False):
        if not isinstance(node, yaml.nodes.ScalarNode):
            raise yaml.constructor.ConstructorError(None, None, 'expected a scalar node, but found {}'.format(node.id), node.start_mark)
        ret_val = node.value if raw else self._ScalarType(node)
        ret_val.lc = yaml.comments.LineCol()
        ret_val.lc.line = node.start_mark.line
        ret_val.lc.col = node.start_mark.column
        ret_val.value = value
        return ret_val

    def construct_scalar(self, node):
        return self._ScalarObject(node, node.value)

    def construct_yaml_null(self, node):
        return self._ScalarObject(node, None)

    def construct_yaml_bool(self, node):
        return self._ScalarObject(node, node.value.lower() == 'true')

    def construct_yaml_int(self, node):
        return self._ScalarObject(node, int(node.value))

    def construct_yaml_float(self, node):
        return self._ScalarObject(node, float(node.value))

    def construct_yaml_map(self, node):
        ret_val = list(super(_LvObjectConstructor, self).construct_yaml_map(node))[0]
        ret_val.value = ret_val
        return ret_val

    def construct_yaml_omap(self, node):
        ret_val = list(super(_LvObjectConstructor, self).construct_yaml_omap(node))[0]
        ret_val.value = ret_val
        return ret_val

    def construct_yaml_seq(self, node):
        ret_val = list(super(_LvObjectConstructor, self).construct_yaml_seq(node))[0]
        ret_val.value = ret_val
        return ret_val