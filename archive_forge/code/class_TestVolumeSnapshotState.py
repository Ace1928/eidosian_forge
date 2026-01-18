import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
class TestVolumeSnapshotState(TestCase):

    def test_volumesnapshotstate_tostring(self):
        self.assertEqual(VolumeSnapshotState.tostring(VolumeSnapshotState.AVAILABLE), 'AVAILABLE')

    def test_volumesnapshotstate_fromstring(self):
        self.assertEqual(VolumeSnapshotState.fromstring('available'), VolumeSnapshotState.AVAILABLE)