from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def GeneratePodSpecFromImages(images):
    """Creates a minimal PodSpec from a list of images.

  Args:
    images: list of images being evaluated.

  Returns:
    PodSpec object in JSON form.
  """
    spec = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': ''}, 'spec': {'containers': [{'image': image} for image in images]}}
    return spec