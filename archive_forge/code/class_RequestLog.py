from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestLog(_messages.Message):
    """Complete log information about a single HTTP request to an App Engine
  application.

  Fields:
    appEngineRelease: App Engine release version.
    appId: Application that handled this request.
    cost: An indication of the relative cost of serving this request.
    endTime: Time when the request finished.
    finished: Whether this request is finished or active.
    first: Whether this is the first RequestLog entry for this request. If an
      active request has several RequestLog entries written to Stackdriver
      Logging, then this field will be set for one of them.
    host: Internet host and port number of the resource being requested.
    httpVersion: HTTP version of request. Example: "HTTP/1.1".
    instanceId: An identifier for the instance that handled the request.
    instanceIndex: If the instance processing this request belongs to a
      manually scaled module, then this is the 0-based index of the instance.
      Otherwise, this value is -1.
    ip: Origin IP address.
    latency: Latency of the request.
    line: A list of log lines emitted by the application while serving this
      request.
    megaCycles: Number of CPU megacycles used to process request.
    method: Request method. Example: "GET", "HEAD", "PUT", "POST", "DELETE".
    moduleId: Module of the application that handled this request.
    nickname: The logged-in user who made the request.Most likely, this is the
      part of the user's email before the @ sign. The field value is the same
      for different requests from the same user, but different users can have
      similar names. This information is also available to the application via
      the App Engine Users API.This field will be populated starting with App
      Engine 1.9.21.
    pendingTime: Time this request spent in the pending request queue.
    referrer: Referrer URL of request.
    requestId: Globally unique identifier for a request, which is based on the
      request start time. Request IDs for requests which started later will
      compare greater as strings than those for requests which started
      earlier.
    resource: Contains the path and query portion of the URL that was
      requested. For example, if the URL was
      "http://example.com/app?name=val", the resource would be
      "/app?name=val". The fragment identifier, which is identified by the #
      character, is not included.
    responseSize: Size in bytes sent back to client by request.
    sourceReference: Source code for the application that handled this
      request. There can be more than one source reference per deployed
      application if source code is distributed among multiple repositories.
    spanId: Stackdriver Trace span identifier for this request.
    startTime: Time when the request started.
    status: HTTP response status code. Example: 200, 404.
    taskName: Task name of the request, in the case of an offline request.
    taskQueueName: Queue name of the request, in the case of an offline
      request.
    traceId: Stackdriver Trace identifier for this request.
    traceSampled: If true, the value in the 'trace_id' field was sampled for
      storage in a trace backend.
    urlMapEntry: File or class that handled the request.
    userAgent: User agent that made the request.
    versionId: Version of the application that handled this request.
    wasLoadingRequest: Whether this was a loading request for the instance.
  """
    appEngineRelease = _messages.StringField(1)
    appId = _messages.StringField(2)
    cost = _messages.FloatField(3)
    endTime = _messages.StringField(4)
    finished = _messages.BooleanField(5)
    first = _messages.BooleanField(6)
    host = _messages.StringField(7)
    httpVersion = _messages.StringField(8)
    instanceId = _messages.StringField(9)
    instanceIndex = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    ip = _messages.StringField(11)
    latency = _messages.StringField(12)
    line = _messages.MessageField('LogLine', 13, repeated=True)
    megaCycles = _messages.IntegerField(14)
    method = _messages.StringField(15)
    moduleId = _messages.StringField(16)
    nickname = _messages.StringField(17)
    pendingTime = _messages.StringField(18)
    referrer = _messages.StringField(19)
    requestId = _messages.StringField(20)
    resource = _messages.StringField(21)
    responseSize = _messages.IntegerField(22)
    sourceReference = _messages.MessageField('SourceReference', 23, repeated=True)
    spanId = _messages.StringField(24)
    startTime = _messages.StringField(25)
    status = _messages.IntegerField(26, variant=_messages.Variant.INT32)
    taskName = _messages.StringField(27)
    taskQueueName = _messages.StringField(28)
    traceId = _messages.StringField(29)
    traceSampled = _messages.BooleanField(30)
    urlMapEntry = _messages.StringField(31)
    userAgent = _messages.StringField(32)
    versionId = _messages.StringField(33)
    wasLoadingRequest = _messages.BooleanField(34)