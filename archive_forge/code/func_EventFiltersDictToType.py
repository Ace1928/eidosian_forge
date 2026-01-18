from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def EventFiltersDictToType(event_filters):
    return next((ef['value'] for ef in event_filters if ef['attribute'] == 'type'), None)