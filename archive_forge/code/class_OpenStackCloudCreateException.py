from openstack import exceptions
class OpenStackCloudCreateException(OpenStackCloudException):

    def __init__(self, resource, resource_id, extra_data=None, **kwargs):
        super(OpenStackCloudCreateException, self).__init__(message='Error creating {resource}: {resource_id}'.format(resource=resource, resource_id=resource_id), extra_data=extra_data, **kwargs)
        self.resource_id = resource_id