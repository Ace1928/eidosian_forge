import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class CircularSubstitutionDefinitionError(Exception):
    pass