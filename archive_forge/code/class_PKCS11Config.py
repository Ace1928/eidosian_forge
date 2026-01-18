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
class PKCS11Config(object):

    def __init__(self, module, slot, label, user_pin):
        self.module = module
        self.slot = slot
        self.label = label
        if user_pin:
            self.user_pin = user_pin