from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import docker_session as v2_session
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from containerregistry.client.v2_2 import docker_session as v2_2_session
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
def Push(image, dest_names, creds, http_obj, session_push_type):
    for dest_name in dest_names:
        with session_push_type(dest_name, creds, http_obj) as push:
            push.upload(image)
            log.CreatedResource(dest_name)