from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
class YAMLObject(metaclass=YAMLObjectMetaclass):
    """
    An object that can dump itself to a YAML stream
    and load itself from a YAML stream.
    """
    __slots__ = ()
    yaml_loader = [Loader, FullLoader, UnsafeLoader]
    yaml_dumper = Dumper
    yaml_tag = None
    yaml_flow_style = None

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Convert a representation node to a Python object.
        """
        return loader.construct_yaml_object(node, cls)

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Convert a Python object to a representation node.
        """
        return dumper.represent_yaml_object(cls.yaml_tag, data, cls, flow_style=cls.yaml_flow_style)