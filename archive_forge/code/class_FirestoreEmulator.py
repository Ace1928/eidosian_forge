from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
class FirestoreEmulator(util.Emulator):
    """Represents the ability to start and route firestore emulator."""

    def Start(self, port):
        args = util.AttrDict({'host_port': {'host': 'localhost', 'port': port}})
        return StartFirestoreEmulator(args, self._GetLogNo())

    @property
    def prefixes(self):
        return ['google.firestore']

    @property
    def service_name(self):
        return FIRESTORE

    @property
    def emulator_title(self):
        return FIRESTORE_TITLE

    @property
    def emulator_component(self):
        return 'cloud-firestore-emulator'