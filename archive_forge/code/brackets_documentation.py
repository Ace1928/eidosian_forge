from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
Return the most recent opening square bracket (if any).