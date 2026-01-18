import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
class TestPL(TestCase):

    def test_obj_track_times(self):
        """
        tests if the object track times  set/get
        """
        gcid = h5p.create(h5p.GROUP_CREATE)
        gcid.set_obj_track_times(False)
        self.assertEqual(False, gcid.get_obj_track_times())
        gcid.set_obj_track_times(True)
        self.assertEqual(True, gcid.get_obj_track_times())
        dcid = h5p.create(h5p.DATASET_CREATE)
        dcid.set_obj_track_times(False)
        self.assertEqual(False, dcid.get_obj_track_times())
        dcid.set_obj_track_times(True)
        self.assertEqual(True, dcid.get_obj_track_times())
        ocid = h5p.create(h5p.OBJECT_CREATE)
        ocid.set_obj_track_times(False)
        self.assertEqual(False, ocid.get_obj_track_times())
        ocid.set_obj_track_times(True)
        self.assertEqual(True, ocid.get_obj_track_times())

    def test_link_creation_tracking(self):
        """
        tests the link creation order set/get
        """
        gcid = h5p.create(h5p.GROUP_CREATE)
        gcid.set_link_creation_order(0)
        self.assertEqual(0, gcid.get_link_creation_order())
        flags = h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED
        gcid.set_link_creation_order(flags)
        self.assertEqual(flags, gcid.get_link_creation_order())
        fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_link_creation_order(flags)
        self.assertEqual(flags, fcpl.get_link_creation_order())

    def test_attr_phase_change(self):
        """
        test the attribute phase change
        """
        cid = h5p.create(h5p.OBJECT_CREATE)
        ret = cid.get_attr_phase_change()
        self.assertEqual((8, 6), ret)
        with self.assertRaises(ValueError):
            cid.set_attr_phase_change(65536, 6)
        cid.set_attr_phase_change(0, 0)
        self.assertEqual((0, 0), cid.get_attr_phase_change())