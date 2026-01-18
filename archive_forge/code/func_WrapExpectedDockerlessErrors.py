from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import re
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_http as v2_docker_http
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from googlecloudsdk.api_lib.container.images import container_analysis_data_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.core.util import times
import six
from six.moves import map
import six.moves.http_client
@contextmanager
def WrapExpectedDockerlessErrors(optional_image_name=None):
    try:
        yield
    except (v2_docker_http.V2DiagnosticException, v2_2_docker_http.V2DiagnosticException) as err:
        if err.status in [six.moves.http_client.UNAUTHORIZED, six.moves.http_client.FORBIDDEN]:
            raise UserRecoverableV2Error('Access denied: {}'.format(optional_image_name or six.text_type(err)))
        elif err.status == six.moves.http_client.NOT_FOUND:
            raise UserRecoverableV2Error('Not found: {}'.format(optional_image_name or six.text_type(err)))
        raise
    except (v2_docker_http.TokenRefreshException, v2_2_docker_http.TokenRefreshException) as err:
        raise TokenRefreshError(six.text_type(err))