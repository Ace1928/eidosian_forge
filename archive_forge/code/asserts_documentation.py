import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import templates
Transforms Assert nodes to Call so they can be handled as functions.