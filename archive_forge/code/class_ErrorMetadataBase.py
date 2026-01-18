import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
class ErrorMetadataBase(object):
    """Container objects attached to exceptions raised in user code.

  This metadata allows re-raising exceptions that occur in generated code, with
  a custom error message that includes a stack trace relative to user-readable
  code from which the generated code originated.
  """
    __slots__ = ('translated_stack', 'cause_message')

    def __init__(self, callsite_tb, cause_metadata, cause_message, source_map, converter_filename):
        translated_stack = _stack_trace_inside_mapped_code(callsite_tb, source_map, converter_filename)
        if cause_metadata is None:
            self.translated_stack = translated_stack
            self.cause_message = cause_message
        else:
            self.translated_stack = cause_metadata.translated_stack + (translated_stack[-1],)
            self.cause_message = cause_metadata.cause_message

    def get_message(self):
        """Returns the message for the underlying exception."""
        lines = []
        lines.append('in user code:')
        lines.append('')
        for frame_info in reversed(self.translated_stack):
            if traceback_utils.is_traceback_filtering_enabled() and (not traceback_utils.include_frame(frame_info.filename)):
                continue
            formatted_line = f'    File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function_name}'
            if frame_info.is_converted:
                formatted_line += '  *'
            elif frame_info.is_allowlisted:
                formatted_line += '  **'
            lines.append(formatted_line)
            if frame_info.code is None:
                code_snippet = '<source unavailable>'
            else:
                code_snippet = frame_info.code.strip()
            lines.append('        {}'.format(code_snippet))
        lines.append('')
        message_lines = self.cause_message.split('\n')
        for i in range(len(message_lines)):
            message_lines[i] = '    ' + message_lines[i]
        lines.extend(message_lines)
        lines.append('')
        return '\n'.join(lines)

    def create_exception(self, source_error):
        """Creates exception from source_error."""
        preferred_type = type(source_error)
        to_ret = None
        if preferred_type.__init__ is Exception.__init__:
            to_ret = preferred_type(self.get_message())
        if preferred_type in KNOWN_STRING_CONSTRUCTOR_ERRORS:
            to_ret = preferred_type(self.get_message())
        elif preferred_type is KeyError:
            to_ret = MultilineMessageKeyError(self.get_message(), self.cause_message)
        if to_ret is not None:
            return to_ret.with_traceback(source_error.__traceback__)

    def to_exception(self, source_error):
        exc = self.create_exception(source_error)
        exc.__suppress_context__ = True
        exc.ag_error_metadata = self
        return exc