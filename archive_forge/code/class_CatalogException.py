from keystoneauth1.exceptions import base
class CatalogException(base.ClientException):
    message = 'Unknown error with service catalog.'