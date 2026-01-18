from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _HasField(field, value):
    return '{} = "{}"'.format(field, value) if value else None