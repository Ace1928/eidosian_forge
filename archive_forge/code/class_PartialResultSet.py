from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartialResultSet(_messages.Message):
    """Partial results from a streaming read or SQL query. Streaming reads and
  SQL queries better tolerate large result sets, large rows, and large values,
  but are a little trickier to consume.

  Fields:
    chunkedValue: If true, then the final value in values is chunked, and must
      be combined with more values from subsequent `PartialResultSet`s to
      obtain a complete field value.
    metadata: Metadata about the result set, such as row type information.
      Only present in the first response.
    resumeToken: Streaming calls might be interrupted for a variety of
      reasons, such as TCP connection loss. If this occurs, the stream of
      results can be resumed by re-sending the original request and including
      `resume_token`. Note that executing any other transaction in the same
      session invalidates the token.
    stats: Query plan and execution statistics for the statement that produced
      this streaming result set. These can be requested by setting
      ExecuteSqlRequest.query_mode and are sent only once with the last
      response in the stream. This field will also be present in the last
      response for DML statements.
    values: A streamed result set consists of a stream of values, which might
      be split into many `PartialResultSet` messages to accommodate large rows
      and/or large values. Every N complete values defines a row, where N is
      equal to the number of entries in metadata.row_type.fields. Most values
      are encoded based on type as described here. It is possible that the
      last value in values is "chunked", meaning that the rest of the value is
      sent in subsequent `PartialResultSet`(s). This is denoted by the
      chunked_value field. Two or more chunked values can be merged to form a
      complete value as follows: * `bool/number/null`: cannot be chunked *
      `string`: concatenate the strings * `list`: concatenate the lists. If
      the last element in a list is a `string`, `list`, or `object`, merge it
      with the first element in the next list by applying these rules
      recursively. * `object`: concatenate the (field name, field value)
      pairs. If a field name is duplicated, then apply these rules recursively
      to merge the field values. Some examples of merging: # Strings are
      concatenated. "foo", "bar" => "foobar" # Lists of non-strings are
      concatenated. [2, 3], [4] => [2, 3, 4] # Lists are concatenated, but the
      last and first elements are merged # because they are strings. ["a",
      "b"], ["c", "d"] => ["a", "bc", "d"] # Lists are concatenated, but the
      last and first elements are merged # because they are lists.
      Recursively, the last and first elements # of the inner lists are merged
      because they are strings. ["a", ["b", "c"]], [["d"], "e"] => ["a", ["b",
      "cd"], "e"] # Non-overlapping object fields are combined. {"a": "1"},
      {"b": "2"} => {"a": "1", "b": 2"} # Overlapping object fields are
      merged. {"a": "1"}, {"a": "2"} => {"a": "12"} # Examples of merging
      objects containing lists of strings. {"a": ["1"]}, {"a": ["2"]} => {"a":
      ["12"]} For a more complete example, suppose a streaming SQL query is
      yielding a result set whose rows contain a single string field. The
      following `PartialResultSet`s might be yielded: { "metadata": { ... }
      "values": ["Hello", "W"] "chunked_value": true "resume_token": "Af65..."
      } { "values": ["orl"] "chunked_value": true } { "values": ["d"]
      "resume_token": "Zx1B..." } This sequence of `PartialResultSet`s encodes
      two rows, one containing the field value `"Hello"`, and a second
      containing the field value `"World" = "W" + "orl" + "d"`. Not all
      `PartialResultSet`s contain a `resume_token`. Execution can only be
      resumed from a previously yielded `resume_token`. For the above sequence
      of `PartialResultSet`s, resuming the query with `"resume_token":
      "Af65..."` will yield results from the `PartialResultSet` with value
      `["orl"]`.
  """
    chunkedValue = _messages.BooleanField(1)
    metadata = _messages.MessageField('ResultSetMetadata', 2)
    resumeToken = _messages.BytesField(3)
    stats = _messages.MessageField('ResultSetStats', 4)
    values = _messages.MessageField('extra_types.JsonValue', 5, repeated=True)