from keystoneauth1.exceptions import base
class EmptyCatalog(EndpointNotFound):
    message = 'The service catalog is empty.'