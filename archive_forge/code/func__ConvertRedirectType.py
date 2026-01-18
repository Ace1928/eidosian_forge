from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def _ConvertRedirectType(redirect_type):
    return {'google-recaptcha': 'GOOGLE_RECAPTCHA', 'external-302': 'EXTERNAL_302'}.get(redirect_type, redirect_type)