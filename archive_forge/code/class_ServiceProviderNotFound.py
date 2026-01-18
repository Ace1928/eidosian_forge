from keystoneauth1.exceptions import base
class ServiceProviderNotFound(base.ClientException):
    """A Service Provider cannot be found."""

    def __init__(self, sp_id):
        self.sp_id = sp_id
        msg = 'The Service Provider %(sp)s could not be found' % {'sp': sp_id}
        super(ServiceProviderNotFound, self).__init__(msg)