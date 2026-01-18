from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.emulators import datastore_util
from googlecloudsdk.command_lib.emulators import pubsub_util
from googlecloudsdk.core.util import files
import six
def WriteRoutesConfig(emulators, output_file):
    """This writes out the routes information to a file.

  The routes will be written as json in the format
  {service1: [route1, route2], service2: [route3, route4]}

  Args:
    emulators: [str], emulators to route the traffic of
    output_file: str, file to write the configuration to
  """
    routes = {name: emulator.prefixes for name, emulator in six.iteritems(emulators)}
    files.WriteFileContents(output_file, json.dumps(routes, indent=2))