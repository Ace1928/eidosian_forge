from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReportedErrorEvent(_messages.Message):
    """An error event which is reported to the Error Reporting system.

  Fields:
    context: Optional. A description of the context in which the error
      occurred.
    eventTime: Optional. Time when the event occurred. If not provided, the
      time when the event was received by the Error Reporting system is used.
      If provided, the time must not exceed the [logs retention
      period](https://cloud.google.com/logging/quotas#logs_retention_periods)
      in the past, or be more than 24 hours in the future. If an invalid time
      is provided, then an error is returned.
    message: Required. The error message. If no `context.reportLocation` is
      provided, the message must contain a header (typically consisting of the
      exception type name and an error message) and an exception stack trace
      in one of the supported programming languages and formats. Supported
      languages are Java, Python, JavaScript, Ruby, C#, PHP, and Go. Supported
      stack trace formats are: * **Java**: Must be the return value of [`Throw
      able.printStackTrace()`](https://docs.oracle.com/javase/7/docs/api/java/
      lang/Throwable.html#printStackTrace%28%29). * **Python**: Must be the
      return value of [`traceback.format_exc()`](https://docs.python.org/2/lib
      rary/traceback.html#traceback.format_exc). * **JavaScript**: Must be the
      value of [`error.stack`](https://github.com/v8/v8/wiki/Stack-Trace-API)
      as returned by V8. * **Ruby**: Must contain frames returned by
      [`Exception.backtrace`](https://ruby-
      doc.org/core-2.2.0/Exception.html#method-i-backtrace). * **C#**: Must be
      the return value of
      [`Exception.ToString()`](https://msdn.microsoft.com/en-
      us/library/system.exception.tostring.aspx). * **PHP**: Must be prefixed
      with `"PHP (Notice|Parse error|Fatal error|Warning): "` and contain the
      result of [`(string)$exception`](https://php.net/manual/en/exception.tos
      tring.php). * **Go**: Must be the return value of
      [`runtime.Stack()`](https://golang.org/pkg/runtime/debug/#Stack).
    serviceContext: Required. The service context in which this error has
      occurred.
  """
    context = _messages.MessageField('ErrorContext', 1)
    eventTime = _messages.StringField(2)
    message = _messages.StringField(3)
    serviceContext = _messages.MessageField('ServiceContext', 4)