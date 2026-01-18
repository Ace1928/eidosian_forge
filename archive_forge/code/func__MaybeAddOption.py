from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
import bootstrapping
from googlecloudsdk.api_lib.iamcredentials import util as iamcred_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce
from googlecloudsdk.core.credentials import store
def _MaybeAddOption(args, name, value):
    if value is None:
        return
    args.append('--{name}={value}'.format(name=name, value=value))