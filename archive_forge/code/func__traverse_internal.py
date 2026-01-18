import enum
import sys
from tensorflow.python.util import tf_inspect
def _traverse_internal(root, visit, stack, path):
    """Internal helper for traverse."""
    if not tf_inspect.isclass(root) and (not tf_inspect.ismodule(root)):
        return
    try:
        children = tf_inspect.getmembers(root)
        if tf_inspect.isclass(root) and issubclass(root, enum.Enum):
            for enum_member in root.__members__.items():
                if enum_member not in children:
                    children.append(enum_member)
            children = sorted(children)
    except ImportError:
        try:
            children = []
            for child_name in root.__all__:
                children.append((child_name, getattr(root, child_name)))
        except AttributeError:
            children = []
    new_stack = stack + [root]
    visit(path, root, children)
    for name, child in children:
        if tf_inspect.ismodule(child) and child.__name__ in sys.builtin_module_names:
            continue
        if any((child is item for item in new_stack)):
            continue
        child_path = path + '.' + name if path else name
        _traverse_internal(child, visit, new_stack, child_path)