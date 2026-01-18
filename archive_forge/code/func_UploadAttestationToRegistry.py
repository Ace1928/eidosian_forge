import base64
import binascii
import collections
import copy
import json
from typing import List, Optional, Text
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_session
from containerregistry.transform.v2_2 import metadata
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_util
from googlecloudsdk.core.exceptions import Error
import httplib2
def UploadAttestationToRegistry(image_url, attestation, use_docker_creds=None, docker_config_dir=None):
    """Uploads a DSSE attestation to the registry.

  Args:
    image_url: The image url of the target image.
    attestation: The attestation referencing the target image in JSON DSSE form.
    use_docker_creds: Whether to use the Docker configuration file for
      authenticating to the registry.
    docker_config_dir: Directory where Docker saves authentication settings.
  """
    http_obj = httplib2.Http()
    target_image = docker_name.Digest(binauthz_util.ReplaceImageUrlScheme(image_url, scheme=''))
    attestation_tag = docker_name.Tag('{}/{}:sha256-{}.att'.format(target_image.registry, target_image.repository, _RemovePrefix(target_image.digest, 'sha256:')))
    creds = None
    if use_docker_creds:
        keychain = docker_creds.DefaultKeychain
        if docker_config_dir:
            keychain.setCustomConfigDir(docker_config_dir)
        creds = keychain.Resolve(docker_name.Registry(target_image.registry))
    if creds is None or isinstance(creds, docker_creds.Anonymous):
        creds = util.CredentialProvider()
    with docker_image.FromRegistry(attestation_tag, creds, http_obj, accepted_mimes=docker_http.SUPPORTED_MANIFEST_MIMES) as v2_2_image:
        if v2_2_image.exists():
            new_image = SigstoreAttestationImage([attestation], v2_2_image)
            docker_session.Push(attestation_tag, creds, http_obj).upload(new_image)
            return
    new_image = SigstoreAttestationImage([attestation])
    docker_session.Push(attestation_tag, creds, http_obj).upload(new_image)