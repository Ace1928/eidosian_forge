import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('oslo_vmware.rw_handles.ImageReadHandle')
@mock.patch.object(image_transfer, 'download_stream_optimized_data')
@mock.patch.object(image_transfer, '_get_vmdk_handle')
def _test_download_stream_optimized_image(self, get_vmdk_handle, download_stream_optimized_data, image_read_handle, container=None, invalid_ova=False):
    image_service = mock.Mock()
    if container:
        image_service.show.return_value = {'container_format': container}
    read_iter = mock.sentinel.read_iter
    image_service.download.return_value = read_iter
    read_handle = mock.sentinel.read_handle
    image_read_handle.return_value = read_handle
    if container == 'ova':
        if invalid_ova:
            get_vmdk_handle.return_value = None
        else:
            vmdk_handle = mock.sentinel.vmdk_handle
            get_vmdk_handle.return_value = vmdk_handle
    imported_vm = mock.sentinel.imported_vm
    download_stream_optimized_data.return_value = imported_vm
    context = mock.sentinel.context
    timeout_secs = mock.sentinel.timeout_secs
    image_id = mock.sentinel.image_id
    session = mock.sentinel.session
    image_size = mock.sentinel.image_size
    host = mock.sentinel.host
    port = mock.sentinel.port
    resource_pool = mock.sentinel.port
    vm_folder = mock.sentinel.vm_folder
    vm_import_spec = mock.sentinel.vm_import_spec
    if container == 'ova' and invalid_ova:
        self.assertRaises(exceptions.ImageTransferException, image_transfer.download_stream_optimized_image, context, timeout_secs, image_service, image_id, session=session, host=host, port=port, resource_pool=resource_pool, vm_folder=vm_folder, vm_import_spec=vm_import_spec, image_size=image_size)
    else:
        ret = image_transfer.download_stream_optimized_image(context, timeout_secs, image_service, image_id, session=session, host=host, port=port, resource_pool=resource_pool, vm_folder=vm_folder, vm_import_spec=vm_import_spec, image_size=image_size)
        self.assertEqual(imported_vm, ret)
        image_service.show.assert_called_once_with(context, image_id)
        image_service.download.assert_called_once_with(context, image_id)
        image_read_handle.assert_called_once_with(read_iter)
        if container == 'ova':
            get_vmdk_handle.assert_called_once_with(read_handle)
            exp_read_handle = vmdk_handle
        else:
            exp_read_handle = read_handle
        download_stream_optimized_data.assert_called_once_with(context, timeout_secs, exp_read_handle, session=session, host=host, port=port, resource_pool=resource_pool, vm_folder=vm_folder, vm_import_spec=vm_import_spec, image_size=image_size)