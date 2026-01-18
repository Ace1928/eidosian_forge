from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import extra_types
from googlecloudsdk.core.util import times
Transform args to a valid cloud event.

  Args:
    event_id: The id of a published event.
    event_type: The event type of a published event.
    event_source: The event source of a published event.
    event_data: The event data of a published event.
    event_attributes: The event attributes of a published event. It can be
      repeated to add more attributes.

  Returns:
    valid CloudEvent.

  