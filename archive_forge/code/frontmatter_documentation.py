import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
Return list of Text nodes for ";"- or ","-separated authornames.