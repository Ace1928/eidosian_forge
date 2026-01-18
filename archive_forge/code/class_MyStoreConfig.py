from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
class MyStoreConfig(object):

    def __init__(self, issuer, store, provider):
        self.issuer = issuer
        self.store = store
        self.provider = provider