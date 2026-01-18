from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def BooleanValue(node_text):
    return node_text.lower() in ('1', 'true')