import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
def _stack_trace_inside_mapped_code(tb, source_map, converter_filename):
    """Summarizes inner traceback frames up to the call to a given function.

  This functions locates the innermost (i.e. most recent) frame that corresponds
  to code that can be mapped by source_map originated from, and returns a
  translated stack trace ending at that frame. If no such frame is found, the
  entire stack trace is summarized.

  For example, the following code:

    def f():
      for i in tf.range(1):
        z = y + i  # z only defined here

  Would generate this traceback:

    <converted code>
        ag__.for_stmt(...)
    <for_stmt>
        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)
    <_known_len_tf_for_stmt>
        _disallow_undefs_into_loop(*init_state)
    <_disallow_undefs_into_loop>
        raise ...

  Which is then processed into:

    <f>
        for i in tf.range(1):
    <for_stmt>
        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)
    <_known_len_tf_for_stmt>
        _disallow_undefs_into_loop(*init_state)
    <_disallow_undefs_into_loop>
        raise ...

  Args:
    tb: traceback.FrameSummary, The traceback corresponding to an error.
      Typically, the output of traceback.Summary.extract(capture_locals=True).
    source_map: Dict[LineLocation, OriginInfo], a source map as created by
      origin_info.create_source_map.
    converter_filename: str, the file path of the converted module. Call frames
      corresponding to this module are elided and their preceding frames are
      marked as allowlisted. Note that frames enclosing converted code are
      dropped using a different mechanism.

  Returns:
    List[FrameInfo]
  """
    result_frames = []
    for filename, line_number, function_name, text in reversed(tb):
        loc = origin_info.LineLocation(filename=filename, lineno=line_number)
        if loc in source_map:
            origin = source_map[loc]
            fi = FrameInfo(filename=origin.loc.filename, lineno=origin.loc.lineno, function_name=origin.function_name, code=origin.source_code_line, is_converted=True, is_allowlisted=False)
            result_frames.append(fi)
            break
        if filename == converter_filename:
            if result_frames:
                prev = result_frames[-1]
                assert not prev.is_converted
                fi = FrameInfo(filename=prev.filename, lineno=prev.lineno, function_name=prev.function_name, code=prev.code, is_converted=False, is_allowlisted=True)
                result_frames[-1] = fi
            continue
        fi = FrameInfo(filename=filename, lineno=line_number, function_name=function_name, code=text, is_converted=False, is_allowlisted=False)
        result_frames.append(fi)
    return tuple(result_frames)