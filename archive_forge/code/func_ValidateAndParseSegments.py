from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def ValidateAndParseSegments(given_segments):
    """Get VideoSegment messages from string of form START1:END1,START2:END2....

  Args:
    given_segments: [str], the list of strings representing the segments.

  Raises:
    SegmentError: if the string is malformed.

  Returns:
    [GoogleCloudVideointelligenceXXXVideoSegment], the messages
      representing the segments or None if no segments are specified.
  """
    if not given_segments:
        return None
    messages = apis.GetMessagesModule(VIDEO_API, VIDEO_API_VERSION)
    segment_msg = messages.GoogleCloudVideointelligenceV1VideoSegment
    segment_messages = []
    segments = [s.split(':') for s in given_segments]
    for segment in segments:
        if len(segment) != 2:
            raise SegmentError(SEGMENT_ERROR_MESSAGE.format(','.join(given_segments), 'Missing start/end segment'))
        start, end = (segment[0], segment[1])
        try:
            start_duration = _ParseSegmentTimestamp(start)
            end_duration = _ParseSegmentTimestamp(end)
        except ValueError as ve:
            raise SegmentError(SEGMENT_ERROR_MESSAGE.format(','.join(given_segments), ve))
        sec_fmt = '{}s'
        segment_messages.append(segment_msg(endTimeOffset=sec_fmt.format(end_duration.total_seconds), startTimeOffset=sec_fmt.format(start_duration.total_seconds)))
    return segment_messages