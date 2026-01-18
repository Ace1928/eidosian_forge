import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
class ImageUtilsHumanQemuTestCase(ImageUtilsHumanRawTestCase):
    _file_format = [('qcow2', dict(file_format='qcow2'))]
    _qcow2_cluster_size = [('65536', dict(cluster_size='65536', exp_cluster_size=65536))]
    _qcow2_encrypted = [('no_encryption', dict(encrypted=None)), ('encrypted', dict(encrypted='yes'))]
    _qcow2_backing_file = [('no_backing_file', dict(backing_file=None)), ('backing_file_path', dict(backing_file='/var/lib/nova/a328c7998805951a_2', exp_backing_file='/var/lib/nova/a328c7998805951a_2')), ('backing_file_path_with_actual_path', dict(backing_file='/var/lib/nova/a328c7998805951a_2 (actual path: /b/3a988059e51a_2)', exp_backing_file='/b/3a988059e51a_2'))]
    _qcow2_backing_file_format = [('no_backing_file_format', dict(backing_file_format=None)), ('backing_file_format', dict(backing_file_format='qcow2', exp_backing_file_format='qcow2'))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._image_name, cls._file_format, cls._virtual_size, cls._disk_size, cls._garbage_before_snapshot, cls._snapshot_count, cls._qcow2_cluster_size, cls._qcow2_encrypted, cls._qcow2_backing_file, cls._qcow2_backing_file_format)

    @mock.patch('debtcollector.deprecate')
    def test_qemu_img_info_human_format(self, mock_deprecate):
        img_info = self._initialize_img_info()
        img_info = img_info + ('cluster_size: %s' % self.cluster_size,)
        if self.backing_file is not None:
            img_info = img_info + ('backing file: %s' % self.backing_file,)
            if self.backing_file_format is not None:
                img_info = img_info + ('backing file format: %s' % self.backing_file_format,)
        if self.encrypted is not None:
            img_info = img_info + ('encrypted: %s' % self.encrypted,)
        if self.garbage_before_snapshot is True:
            img_info = img_info + ('blah BLAH: bb',)
        if self.snapshot_count is not None:
            img_info = self._insert_snapshots(img_info)
        if self.garbage_before_snapshot is False:
            img_info = img_info + ('junk stuff: bbb',)
        example_output = '\n'.join(img_info)
        warnings.simplefilter('always', FutureWarning)
        image_info = imageutils.QemuImgInfo(example_output)
        mock_deprecate.assert_called()
        self._base_validation(image_info)
        self.assertEqual(image_info.cluster_size, self.exp_cluster_size)
        if self.backing_file is not None:
            self.assertEqual(image_info.backing_file, self.exp_backing_file)
            if self.backing_file_format is not None:
                self.assertEqual(image_info.backing_file_format, self.exp_backing_file_format)
        if self.encrypted is not None:
            self.assertEqual(image_info.encrypted, self.encrypted)