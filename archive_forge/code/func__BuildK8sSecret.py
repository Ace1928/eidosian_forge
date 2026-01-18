from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
def _BuildK8sSecret(secret_name, secrets, namespace):
    """Turn a map of SecretManager responses into a k8s secret."""
    data_value = RUN_MESSAGE_MODULE.Secret.DataValue(additionalProperties=[])
    k8s_secret = RUN_MESSAGE_MODULE.Secret(metadata=RUN_MESSAGE_MODULE.ObjectMeta(name=secret_name, namespace=namespace), data=data_value)
    for version, secret in secrets.items():
        k8s_secret.data.additionalProperties.append(RUN_MESSAGE_MODULE.Secret.DataValue.AdditionalProperty(key=version, value=secret.payload.data))
    d = encoding_helper.MessageToDict(k8s_secret)
    d['apiVersion'] = 'v1'
    d['kind'] = 'Secret'
    return d