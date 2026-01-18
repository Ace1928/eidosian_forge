import abc
from openstack import exceptions
from openstack import proxy
class BaseBlockStorageProxy(proxy.Proxy, metaclass=abc.ABCMeta):

    def create_image(self, name, volume, allow_duplicates, container_format, disk_format, wait, timeout):
        if not disk_format:
            disk_format = self._connection.config.config['image_format']
        if not container_format:
            container_format = 'bare'
        if 'id' in volume:
            volume_id = volume['id']
        else:
            volume_obj = self.get_volume(volume)
            if not volume_obj:
                raise exceptions.SDKException('Volume {volume} given to create_image could not be found'.format(volume=volume))
            volume_id = volume_obj['id']
        data = self.post('/volumes/{id}/action'.format(id=volume_id), json={'os-volume_upload_image': {'force': allow_duplicates, 'image_name': name, 'container_format': container_format, 'disk_format': disk_format}})
        response = self._connection._get_and_munchify('os-volume_upload_image', data)
        return self._connection.image._existing_image(id=response['image_id'])