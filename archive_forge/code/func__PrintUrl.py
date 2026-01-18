from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import os.path
import signal
import subprocess
import sys
import threading
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def _PrintUrl(service_name, events_port, stop):
    """Read the local url of a service from the event stream and print it.

  Read the event stream api and find the portForward events. Print the local
  url as determined from the portFoward events. This function will continuously
  listen to the event stream and print out all local urls until eitherthe event
  stream connection closes or the stop event is set.

  Args:
    service_name: Name of the service.
    events_port: Port number of the skaffold events stream api.
    stop: threading.Event event.
  """
    try:
        with contextlib.closing(_OpenEventStreamRetry(events_port, stop)) as response:
            for port in GetServiceLocalPort(response, service_name):
                if stop.is_set():
                    return
                con = console_attr.GetConsoleAttr()
                msg = 'Service URL: {bold}{url}{normal}'.format(bold=con.GetFontCode(bold=True), url='http://localhost:%s/' % port, normal=con.GetFontCode())
                stop.wait(1)
                log.status.Print(con.Colorize(msg, color='blue'))
    except StopThreadError:
        return