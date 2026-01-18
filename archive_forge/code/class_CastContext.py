import abc
from typing import Any, List, Optional, Sequence, Iterator
from typing_extensions import Protocol
from typing_extensions import runtime_checkable
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class CastContext:
    """Contains context info and rules for casting values to a TypeSpec."""