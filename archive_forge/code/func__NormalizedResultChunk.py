from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apigee import request
@classmethod
def _NormalizedResultChunk(cls, result_chunk):
    """Returns a list of the results in `result_chunk`."""
    if cls._list_container is None:
        return result_chunk
    try:
        return result_chunk[cls._list_container]
    except KeyError:
        failure_info = (cls, cls._list_container, result_chunk)
        raise AssertionError("%s specifies a _list_container %r that's not present in API responses.\nResponse: %r" % failure_info)
    except (IndexError, TypeError):
        error = '%s specifies a _list_container, implying that the API response should be a JSON object, but received something else instead: %r' % (cls, result_chunk)
        raise AssertionError(error)